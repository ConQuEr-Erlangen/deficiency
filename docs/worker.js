// Web worker: loads Pyodide + block_jacobi_mp.py once, then runs
// quick_deficiency_indices for every {id, poly} message it receives.

const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.2/full/";
const MODULE_URL = new URL("block_jacobi_mp.py", self.location).href;

importScripts(PYODIDE_URL + "pyodide.js");

const RUNNER = `
import io, json, warnings
import block_jacobi_mp as bj

def run(poly):
    buf = io.StringIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = bj.quick_deficiency_indices(poly, return_result=True, color=False, file=buf)
    states = [r["state"] for r in res]
    npl = nmi = None
    if all(s == 0 for s in states):
        npl, nmi = res[0]["n"], res[1]["n"]
        kind = "good" if npl == nmi == 0 else ("soso" if npl == nmi else "bad")
    elif 2 in states:
        kind = "nonhermitian"
    else:
        kind = "error"
    return json.dumps({
        "display": bj._display(poly),
        "kind": kind, "nplus": npl, "nminus": nmi,
        "output": buf.getvalue(),
        "warnings": [str(w.message) for w in caught],
        "details": res,
        "real": res[0] is res[1],   # real coefficients: n- taken over from n+
    }, default=str)

def weyl(poly):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        text = bj.weyl_symmetrize(poly)
    return json.dumps({"text": text, "hermitian": bj.is_hermitian(text),
                       "warnings": [str(w.message) for w in caught]})

def herm(poly):
    text = bj.hermitian_part(poly)
    return json.dumps({"text": text, "changed": text != poly})
`;

async function init() {
  self.postMessage({ type: "status", text: "Loading Python runtime…" });
  const pyodide = await loadPyodide({ indexURL: PYODIDE_URL });

  self.postMessage({ type: "status", text: "Loading sympy and mpmath…" });
  await pyodide.loadPackage(["sympy", "mpmath"]);
  try {
    await pyodide.loadPackage("gmpy2");    // optional speed-up, module falls back to int
  } catch (_) {}

  self.postMessage({ type: "status", text: "Loading block_jacobi_mp.py…" });
  const resp = await fetch(MODULE_URL);
  if (!resp.ok) throw new Error(`could not fetch ${MODULE_URL} (${resp.status})`);
  pyodide.FS.writeFile("block_jacobi_mp.py", await resp.text());
  pyodide.runPython(RUNNER);
  return {
    run: (poly) => JSON.parse(pyodide.globals.get("run")(poly)),
    weyl: (poly) => JSON.parse(pyodide.globals.get("weyl")(poly)),
    herm: (poly) => JSON.parse(pyodide.globals.get("herm")(poly)),
  };
}

const ready = init().then(
  (ops) => { self.postMessage({ type: "ready" }); return ops; },
  (err) => { self.postMessage({ type: "fatal", message: String(err) }); throw err; },
);

// Messages: {id, op: "run" | "weyl" | "herm", poly}
self.onmessage = async ({ data: { id, op, poly } }) => {
  const ops = await ready;
  try {
    self.postMessage({ type: "result", id, result: ops[op](poly) });
  } catch (err) {
    // Python exceptions: keep only the last line ("ValueError: ...") for the user
    const msg = String(err.message || err).trim().split("\n").pop();
    self.postMessage({ type: "error", id, message: msg });
  }
};
