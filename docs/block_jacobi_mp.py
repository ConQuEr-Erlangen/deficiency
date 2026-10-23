"""
Deficiency indices of bosonic polynomials via block-Jacobi matrices.

Pure-Python port of ``block-jacobi.wl`` (``DeficiencyIndices``,
``AdaptiveDeficiencyIndices``, ``DeficiencyPlot`` and
``QuickDeficiencyIndices``), built on sympy, mpmath and gmpy2 so that it runs
under Pyodide.  It is a translation of ``block_jacobi.py``, which uses
python-flint; the public API is the same (except for ``error_control``, see
below).

Input
-----
Polynomials are given symbolically in the noncommuting symbols ``a``, ``adag``
(with ``[a, adag] = 1``) or ``x``/``q``, ``p`` (with ``[x, p] = i``).  Either as a
string::

    quick_deficiency_indices("x^4 - 1/10*p^2 + sqrt(2)*(x*p + p*x)")

or as a sympy expression built from the symbols exported by this module::

    from block_jacobi_mp import a, adag, x, p
    quick_deficiency_indices(x**4 + sympy.Rational(1, 10)*p**2)

All coefficients are kept *exact* (sympy integers, rationals, radicals, pi, ...)
and are only evaluated at the working precision of the current run, so they
always carry full precision.  Decimal literals in strings (``0.1``) are parsed
as exact rationals.  Inexact sympy ``Float`` coefficients are accepted, but they
only carry their own precision; if the algorithm needs more, it stops with
state ``+1`` (like Mathematica does for machine-precision input).

Precision control
-----------------
All numerics run in mpmath multiprecision floating point (with the gmpy2
backend when it is installed).  Every computation of the Weyl radii is done
twice, at ``bits`` and at ``bits + guard`` bits, each time re-evaluating the
exact coefficients at that precision.  The error of the more precise run is
estimated by twice the difference of the two runs plus one unit in the last
place of the lower one, and the radii are carried on as balls (midpoint,
radius).  All convergence decisions are taken on these balls with three-valued
logic: if a ball straddles a threshold, the precision is too low to decide and
the run reports state -3; the adaptive drivers then double the precision.
Garbage produced by cancellation or ill-conditioning at the lower precision
shows up as a large difference and is caught the same way.

Error codes (identical to the Mathematica version)
--------------------------------------------------
 0: Success
-1: not converged yet
-2: radia increase -> precision too low
-3: radia not resolved accurately enough -> precision too low
-4: deficiency indices do not sum up to d (convergence issues)
-5: imaginary radia -> precision too low (cannot occur here: radii are
    computed from Hermitian matrices; kept for compatibility)
-6: singular matrix while calculating P[z] -> increase start index
-7: radia not bounded away from zero -> precision too low
-8: error whilst calculating polynomials, not due to det 0 -> precision too low
+1: coefficient list does not have sufficient precision
+2: coefficient list not hermitian
"""

from __future__ import annotations

import itertools
import math
import sys
import warnings

import mpmath
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication,
    parse_expr,
    rationalize,
    split_symbols,
    standard_transformations,
)

try:                                   # exact big-integer products; optional
    from gmpy2 import mpz as _mpz
except ImportError:                    # pragma: no cover
    _mpz = int

__all__ = [
    "a", "adag", "x", "p", "q",
    "MACHINE_PRECISION",
    "parse_polynomial", "normal_order", "coefficient_matrix", "dagger",
    "hermitianify", "is_hermitian", "hermitian_part", "weyl_symmetrize", "deficiency_indices", "adaptive_deficiency_indices",
    "weyl_radii", "deficiency_plot", "quick_deficiency_indices",
    "PrecisionError",
]

# Private mpmath context, so the global mpmath.mp precision is never touched.
mp = mpmath.MPContext()

# Mathematica's MachinePrecision, in decimal digits (= 53 bits).
MACHINE_PRECISION = 53 * math.log10(2)

a, adag = sp.symbols("a adag", commutative=False)
x, p, q = sp.symbols("x p q", commutative=False)


class PrecisionError(RuntimeError):
    """Raised when the requested accuracy cannot be reached within the precision limits."""


def _digits_to_bits(digits):
    return max(2, int(round(digits * math.log2(10))))


def _workprec(bits):
    return mp.workprec(bits)


# ---------------------------------------------------------------------------
# Symbolic part: parsing, normal ordering, coefficient matrix
# ---------------------------------------------------------------------------

_PARSE_LOCALS = {"a": a, "adag": adag, "ad": adag, "x": x, "q": q, "p": p,
                 "I": sp.I, "i": sp.I,
                 "comm": lambda A, B: A*B - B*A}


_TRANSFORMATIONS = standard_transformations + (convert_xor, rationalize,
                                               split_symbols, implicit_multiplication)


def parse_polynomial(s):
    """Parse a string into a sympy polynomial in a, adag (or x, q, p).

    ``^`` is accepted for powers and decimal literals become exact rationals.
    ``comm(A, B)`` denotes the commutator ``A*B - B*A``.  Multiplication may be
    implicit: ``2x^2 p``, ``xp`` and ``(x+p)(x-p)`` are understood.
    """
    return parse_expr(s, local_dict=dict(_PARSE_LOCALS), transformations=_TRANSFORMATIONS)


def _as_expr(poly):
    if isinstance(poly, str):
        return parse_polynomial(poly)
    return sp.sympify(poly)


# A normal-ordered polynomial is a dict {(k, l): c} meaning sum c adag^k a^l.

def _no_add(P, Q):
    R = dict(P)
    for key, c in Q.items():
        R[key] = R.get(key, 0) + c
    return R


def _no_mul(P, Q):
    # a^l adag^m = sum_r C(l,r) C(m,r) r! adag^(m-r) a^(l-r)
    R = {}
    for (k1, l1), c1 in P.items():
        for (k2, l2), c2 in Q.items():
            for r in range(min(l1, k2) + 1):
                w = math.comb(l1, r) * math.comb(k2, r) * math.factorial(r)
                key = (k1 + k2 - r, l1 + l2 - r)
                R[key] = R.get(key, 0) + w * c1 * c2
    return R


def _no_pow(P, n):
    R = {(0, 0): sp.Integer(1)}
    while n:
        if n & 1:
            R = _no_mul(R, P)
        P = _no_mul(P, P)
        n >>= 1
    return R


_SQ2 = sp.sqrt(2)
_GENERATORS = {
    a: {(0, 1): sp.Integer(1)},
    adag: {(1, 0): sp.Integer(1)},
    # x = (a + adag)/sqrt2,  p = (a - adag)/(sqrt2 i)
    x: {(0, 1): 1 / _SQ2, (1, 0): 1 / _SQ2},
    q: {(0, 1): 1 / _SQ2, (1, 0): 1 / _SQ2},
    p: {(0, 1): -sp.I / _SQ2, (1, 0): sp.I / _SQ2},
}


def _to_no(expr):
    if expr in _GENERATORS:
        return dict(_GENERATORS[expr])
    if expr.is_commutative:
        if expr.free_symbols:
            raise ValueError(f"coefficients must be numeric, got free symbols {expr.free_symbols}")
        return {(0, 0): expr}
    if isinstance(expr, sp.Add):
        R = {}
        for t in expr.args:
            R = _no_add(R, _to_no(t))
        return R
    if isinstance(expr, sp.Mul):
        R = {(0, 0): sp.Integer(1)}
        for f in expr.args:          # sympy keeps the order of noncommutative factors
            R = _no_mul(R, _to_no(f))
        return R
    if isinstance(expr, sp.Pow):
        n = expr.exp
        if not (n.is_Integer and n >= 0):
            raise ValueError(f"only nonnegative integer powers of operators allowed: {expr}")
        return _no_pow(_to_no(expr.base), int(n))
    raise ValueError(f"cannot interpret {expr!r} as a polynomial in a, adag, x, p")


def _is_zero(c):
    c = sp.expand(c)
    if c == 0:
        return True
    z = c.is_zero
    if z is not None:
        return z
    if sp.simplify(c) == 0:
        return True
    return abs(complex(sp.N(c, 60))) < 1e-50


def normal_order(poly):
    """Normal-ordered form {(k, l): c} of ``poly``, meaning sum_c c adag^k a^l."""
    R = _to_no(_as_expr(poly))
    return {key: sp.expand(c) for key, c in R.items() if not _is_zero(c)}


def coefficient_matrix(poly):
    """Square sympy matrix C with C[k, l] = coefficient of adag^k a^l (normal ordered).

    Equivalent of ``Coefficients[toBosonicPoly@poly]``.
    """
    R = normal_order(poly)
    if not R:
        raise ValueError("polynomial is zero")
    D = 1 + max(max(k, l) for k, l in R)
    C = sp.zeros(D, D)
    for (k, l), c in R.items():
        C[k, l] = c
    return C


def dagger(poly):
    """Formal adjoint of a polynomial expression (HermConj)."""
    e = _as_expr(poly)
    if e == a:
        return adag
    if e == adag:
        return a
    if e in (x, p, q):
        return e
    if e.is_commutative:
        return sp.conjugate(e)
    if isinstance(e, sp.Add):
        return sp.Add(*[dagger(t) for t in e.args])
    if isinstance(e, sp.Mul):
        return sp.Mul(*[dagger(f) for f in reversed(e.args)])
    if isinstance(e, sp.Pow):
        return dagger(e.base) ** e.exp
    raise ValueError(f"cannot take adjoint of {e!r}")


def hermitianify(poly):
    """Hermitian part (poly + poly^dagger)/2, expanded."""
    e = _as_expr(poly)
    return sp.expand((e + dagger(e)) / 2)


def is_hermitian(poly):
    """True if the operator equals its adjoint."""
    return _is_hermitian(coefficient_matrix(poly)) if normal_order(poly) else True


def hermitian_part(poly):
    """(H + H^dagger)/2 as an input string; Hermitian input is returned unchanged.

    The ordering of the factors is kept.  Applied to a Weyl-symmetrized
    polynomial this gives the Weyl operator of the real part of the symbol.
    """
    if is_hermitian(poly):
        return poly if isinstance(poly, str) else sp.sstr(poly).replace("**", "^")
    e = _as_expr(poly)
    return sp.sstr(sp.expand((e + dagger(e)) / 2)).replace("**", "^")


def _no_commutator(A, B):
    raise ValueError("comm() cannot be used in a classical polynomial to be Weyl-symmetrized")


def _word(u, v, positions, n):
    """Monomial with u at ``positions`` and v elsewhere, runs written as powers."""
    letters = [u if i in positions else v for i in range(n)]
    runs = []
    for s in letters:
        if runs and runs[-1][0] == s:
            runs[-1][1] += 1
        else:
            runs.append([s, 1])
    return "*".join(s if m == 1 else f"{s}^{m}" for s, m in runs)


def _coefficient_prefix(c):
    if c == 1:
        return ""
    if c == -1:
        return "-"
    s = sp.sstr(c).replace("**", "^")
    return (f"({s})" if isinstance(c, sp.Add) else s) + "*"


def weyl_symmetrize(poly):
    """Weyl-ordered operator of a classical polynomial, as an input string.

    The variables in ``poly`` are treated as commuting, and every monomial
    u^k v^l is replaced by the average of all orderings of its k+l factors,
    e.g. ``x^2 p`` -> ``1/3*(x^2*p + x*p*x + p*x^2)``.  Only polynomials in p
    and either x or q are accepted; input containing a or adag, or both x and
    q, is returned unchanged with a warning.
    """
    cl = {name: sp.Symbol(name) for name in ("a", "adag", "x", "q", "p")}
    local = {**_PARSE_LOCALS, **cl, "ad": cl["adag"], "comm": _no_commutator}
    e = sp.expand(parse_expr(poly, local_dict=local, transformations=_TRANSFORMATIONS))
    unknown = e.free_symbols - set(cl.values())
    if unknown:
        raise ValueError(f"unknown symbols {unknown}")
    used = {s.name for s in e.free_symbols}
    unchanged = poly if isinstance(poly, str) else sp.sstr(poly).replace("**", "^")
    if used & {"a", "adag"}:
        warnings.warn("Weyl symmetrization should only be applied onto polynomials in x and p. "
                      "Returning original polynomial")
        return unchanged
    if {"x", "q"} <= used:
        warnings.warn("Use either x or q. Returning original polynomial")
        return unchanged
    u = "q" if "q" in used else "x"
    v = "p"
    terms = sp.Poly(e, cl[u], cl[v]).terms()      # sorted by descending degree
    if not terms:
        return "0"
    parts = []
    for (k, l), c in terms:
        n = k + l
        if n == 0:
            parts.append(sp.sstr(c).replace("**", "^"))
            continue
        words = [_word(u, v, set(pos), n) for pos in itertools.combinations(range(n), k)]
        if len(words) == 1:
            parts.append(_coefficient_prefix(c) + words[0])
        else:
            parts.append(_coefficient_prefix(c / len(words)) + "(" + " + ".join(words) + ")")
    return " + ".join(parts).replace("+ -", "- ")


def _as_coefficients(obj):
    if isinstance(obj, sp.MatrixBase):
        return sp.Matrix(obj)
    if isinstance(obj, (list, tuple)):
        return sp.Matrix(obj)
    return coefficient_matrix(obj)


def _is_hermitian(C):
    return all(_is_zero(C[k, l] - sp.conjugate(C[l, k]))
               for k in range(C.rows) for l in range(k, C.cols))


def _is_real(C):
    """FreeQ[coefficients, _Complex]: the operator commutes with complex conjugation."""
    return not C.has(sp.I)


def _nonzero_positions(C):
    return [(k, l) for k in range(C.rows) for l in range(C.cols) if C[k, l] != 0]


def _max_shift(C):
    return max(abs(k - l) for k, l in _nonzero_positions(C))


def _max_power(C):
    return max(k + l for k, l in _nonzero_positions(C))


def _by_diagonal(C):
    groups = {}
    for k, l in _nonzero_positions(C):
        groups.setdefault(l - k, []).append((k, l))
    return {s: sorted(v) for s, v in groups.items()}


def _dominant_n(C):
    """getDominantN: index from which every diagonal is in its asymptotic regime
    and the dominant diagonal dominates the others."""
    def n_for(lead, other):
        (km, lm), (k, l) = lead, other
        ratio = abs(complex(sp.N(C[k, l]))) / abs(complex(sp.N(C[km, lm])))
        return math.ceil(ratio ** (2 / ((km + lm) - (k + l))))

    groups = _by_diagonal(C)
    best = 1
    for pos in groups.values():                   # leading term of each diagonal
        for kl in pos[:-1]:
            best = max(best, n_for(pos[-1], kl))
    leads = sorted((pos[-1] for pos in groups.values()), key=lambda kl: (sum(kl), kl))
    for kl in leads[:-1]:                         # leading term of the dominant diagonal
        if sum(kl) != sum(leads[-1]):
            best = max(best, n_for(leads[-1], kl))
    return best


def _rescaling(C):
    """getRescaling: 1/|leading coefficient of the lowering-most diagonal| (exact)."""
    groups = _by_diagonal(C)
    k, l = groups[max(groups)][-1]
    return 1 / sp.Abs(C[k, l])


def _scale_test(C):
    r = float(_rescaling(C))
    return _dominant_n(C) > 10 or r > 10 or 1 / r > 10


def _coefficient_precision_bits(C):
    """Precision of the coefficients in bits (inf if all exact), like Precision[]."""
    floats = [f for c in C for f in sp.sympify(c).atoms(sp.Float)]
    return min((f._prec for f in floats), default=math.inf)


# ---------------------------------------------------------------------------
# Exact -> mpmath conversion
# ---------------------------------------------------------------------------

def _float_to_mpf(f):
    """sympy Float (or Integer) -> mpf, correctly rounded to the working precision."""
    if isinstance(f, sp.Float):
        return mp.mpf(f._mpf_)
    return mp.mpf(int(f)) if f.is_Integer else mp.mpf(int(f.p)) / int(f.q)


def _to_mpc(e):
    """Evaluate a numeric sympy expression as an mpc at the current working precision.

    Integers and rationals are rounded once; sympy Floats are taken at their
    exact binary value; everything else is evaluated by sympy's adaptive
    ``evalf`` with 20 guard digits, which guarantees the requested accuracy.
    """
    if isinstance(e, (int, float, complex)) and not isinstance(e, bool):
        return mp.mpc(e)                    # python numbers are taken as exact binary values
    e = sp.sympify(e)
    if e.is_Integer:
        return mp.mpc(int(e))
    if e.is_Rational:
        return mp.mpc(mp.mpf(int(e.p)) / int(e.q))
    if isinstance(e, sp.Float):
        return mp.mpc(_float_to_mpf(e))
    if e is sp.I:
        return mp.mpc(0, 1)
    digits = int(mp.prec / math.log2(10)) + 20
    v = sp.N(e, digits)
    if v.free_symbols:
        raise ValueError(f"coefficient {e} is not numeric")
    re_, im_ = v.as_real_imag()
    return mp.mpc(_float_to_mpf(sp.Float(re_, digits)), _float_to_mpf(sp.Float(im_, digits)))


# ---------------------------------------------------------------------------
# Small dense complex linear algebra (lists of lists of mpc)
# ---------------------------------------------------------------------------

def _identity(d):
    return [[mp.mpc(1) if i == j else mp.mpc(0) for j in range(d)] for i in range(d)]


def _zeros(d):
    return [[mp.mpc(0)] * d for _ in range(d)]


def _matmul(A, B):
    Bt = list(zip(*B))
    return [[mp.fdot(row, col) for col in Bt] for row in A]


def _adjoint_mul(A, B):
    """A^dagger B."""
    At = list(zip(*A))
    Bt = list(zip(*B))
    return [[mp.fdot(bj, ai, conjugate=True) for bj in Bt] for ai in At]


def _matsub(A, B):
    return [[u - v for u, v in zip(ra, rb)] for ra, rb in zip(A, B)]


def _matadd(A, B):
    return [[u + v for u, v in zip(ra, rb)] for ra, rb in zip(A, B)]


def _scaled_identity_minus(z, B):
    """z*1 - B."""
    return [[(z if i == j else 0) - B[i][j] for j in range(len(B))] for i in range(len(B))]


class _SingularBlock(ZeroDivisionError):
    def __init__(self, n):
        super().__init__(f"A block {n} is numerically singular")
        self.n = n


def _solve(A, R):
    """Solve A X = R by Gaussian elimination with partial pivoting.

    Raises ZeroDivisionError if a pivot is zero to working precision relative
    to the size of A, i.e. A is numerically singular.
    """
    d = len(A)
    A = [list(row) for row in A]
    R = [list(row) for row in R]
    norm = max(mp.fsum(abs(v) for v in row) for row in A)
    tol = norm * d * mp.eps
    for j in range(d):
        piv = max(range(j, d), key=lambda i: abs(A[i][j]))
        if not abs(A[piv][j]) > tol:
            raise ZeroDivisionError("matrix is numerically singular")
        A[j], A[piv] = A[piv], A[j]
        R[j], R[piv] = R[piv], R[j]
        inv = 1 / A[j][j]
        for i in range(j + 1, d):
            f = A[i][j] * inv
            if f:
                A[i] = [u - f * v for u, v in zip(A[i], A[j])]
                R[i] = [u - f * v for u, v in zip(R[i], R[j])]
    X = [None] * d
    for j in reversed(range(d)):
        row = R[j]
        for k in range(j + 1, d):
            if A[j][k]:
                row = [u - A[j][k] * v for u, v in zip(row, X[k])]
        X[j] = [u / A[j][j] for u in row]
    return X


def _det(A):
    try:
        return mp.det(mp.matrix(A))
    except ZeroDivisionError:
        return mp.mpf(0)


# ---------------------------------------------------------------------------
# Block Jacobi matrices and orthogonal polynomials
# ---------------------------------------------------------------------------

def _rising(lo, hi):
    """(lo+1)(lo+2)...hi as an exact integer."""
    return math.prod(range(lo + 1, hi + 1), start=_mpz(1))


class _Blocks:
    """Blocks A[n], B[n] of the block-Jacobi form of H (block size d).

    A[n][i, j] = <dN+i|H|d(N+1)+j>,  B[n][i, j] = <dN+i|H|dN+j>,  N = n + start.
    Same as ABlock/BBlock in the Mathematica code, computed directly from the
    matrix elements <m|adag^k a^l|n>.  Must be created inside the working
    precision it is used at.
    """

    def __init__(self, C, d, start):
        self.d, self.start = d, start
        self.terms = {}
        self._cache = {}
        for k in range(C.rows):
            for l in range(C.cols):
                if C[k, l] != 0:
                    self.terms.setdefault(l - k, []).append((k, l, _to_mpc(C[k, l])))

    def element(self, row, col):
        terms = []
        for k, l, c in self.terms.get(col - row, ()):
            if col < l:
                continue
            base = col - l
            # <base+k| adag^k a^l |col> = sqrt(col!/base! * (base+k)!/base!)
            m = _rising(base, col) * _rising(base, base + k)
            terms.append(c * mp.sqrt(int(m)))
        return mp.fsum(terms)

    def _block(self, kind, n):
        key = (kind, n)
        if key not in self._cache:
            d, N = self.d, n + self.start
            off = d if kind == "A" else 0
            self._cache[key] = [[self.element(d * N + i, d * N + off + j) for j in range(d)]
                                for i in range(d)]
        return self._cache[key]

    def A(self, n):
        return self._block("A", n)

    def B(self, n):
        return self._block("B", n)


def _p_polynomials(z, blocks, length):
    """PFn: matrix orthogonal polynomials P_0..P_length evaluated at z."""
    d = blocks.d
    prev, cur, A_prev = None, _identity(d), None
    P = [cur]
    for n in range(length):
        A = blocks.A(n)
        rhs = _matmul(_scaled_identity_minus(z, blocks.B(n)), cur)
        if A_prev is not None:
            rhs = _matsub(rhs, _adjoint_mul(A_prev, prev))
        try:
            nxt = _solve(A, rhs)
        except ZeroDivisionError:
            raise _SingularBlock(n) from None
        P.append(nxt)
        prev, cur, A_prev = cur, nxt, A
    return P


def _singular_shift(blocks, upto):
    """checkABlocks: 1 + largest n <= upto with |det A[n]| < 1, or None."""
    bad = [n for n in range(upto + 1) if abs(_det(blocks.A(n))) < 1]
    return (max(bad) + 1) if bad else None


def _hermitian_eigenvalues(M):
    """Eigenvalues (ascending) of the Hermitian part of M, as mpf.

    Solved with 10 extra bits; the remaining error is caught by the
    two-precision comparison in ``_compute_radii``.  Returns None if the
    eigenvalue iteration fails.
    """
    d = len(M)
    if d == 1:
        return [M[0][0].real]
    H = [[None] * d for _ in range(d)]
    real = True
    for i in range(d):
        H[i][i] = M[i][i].real
        for j in range(i + 1, d):
            v = (M[i][j] + M[j][i].conjugate()) / 2
            if v.imag:
                real = False
            H[i][j], H[j][i] = v, v.conjugate()
    if real:                        # mpmath uses the (faster) real symmetric solver
        H = [[v.real if isinstance(v, mp.mpc) else v for v in row] for row in H]
    try:
        with mp.extraprec(10):
            ev = mp.eigh(mp.matrix(H), eigvals_only=True)
    except Exception:               # no convergence: treat as unresolved
        return None
    return sorted(+mp.mpf(e) for e in ev)


def _weyl_radii(z, P):
    """weylRadius: eigenvalues of (2|Im z| sum_{k<=N} P_k^dag P_k)^-1 for every N.

    Returns radii[N][i] (mpf), each row sorted in descending order like
    Mathematica's Eigenvalues; nan where the eigenvalue is not positive.
    """
    d = len(P[0])
    factor = 2 * abs(z.imag)
    S = _zeros(d)
    radii = []
    for Pk in P:
        S = _matadd(S, _adjoint_mul(Pk, Pk))
        lams = _hermitian_eigenvalues([[factor * v for v in row] for row in S])
        if lams is None:
            radii.append([mp.nan] * d)
        else:                       # ascending lambda -> descending radius
            radii.append([1 / lam if lam > 0 else mp.nan for lam in lams])
    return radii


def _run_radii(z, C, d, start, length, bits):
    """Run PFn + weylRadius at ``bits`` precision.

    Returns ("ok", radii) or (state, extra_info) on failure.
    """
    with _workprec(bits):
        zb = _to_mpc(z)
        blocks = _Blocks(C, d, start)
        try:
            P = _p_polynomials(zb, blocks, length)
        except _SingularBlock as exc:
            shift = _singular_shift(blocks, exc.n)
            if shift is None:
                return -8, {}
            return -6, {"deltaShift": shift}
        return "ok", _weyl_radii(zb, P)


# ---------------------------------------------------------------------------
# Balls: midpoint + error estimate
# ---------------------------------------------------------------------------

class _Ball:
    """Real number known as mid +- rad.  Operations propagate rad (first order
    exact bounds) and add one ulp of the working precision for rounding."""

    __slots__ = ("mid", "rad")

    def __init__(self, mid, rad=0):
        self.mid = mp.mpf(mid)
        self.rad = mp.mpf(rad)

    @classmethod
    def nan(cls):
        return cls(mp.nan, mp.inf)

    @classmethod
    def _make(cls, mid, rad):
        return cls(mid, rad + abs(mid) * mp.eps)

    def is_finite(self):
        return mp.isfinite(self.mid) and mp.isfinite(self.rad)

    def lower(self):
        return self.mid - self.rad

    def upper(self):
        return self.mid + self.rad

    @staticmethod
    def _coerce(o):
        return o if isinstance(o, _Ball) else _Ball(o)

    def __add__(self, o):
        o = self._coerce(o)
        return _Ball._make(self.mid + o.mid, self.rad + o.rad)

    __radd__ = __add__

    def __neg__(self):
        return _Ball(-self.mid, self.rad)

    def __sub__(self, o):
        return self + (-self._coerce(o))

    def __rsub__(self, o):
        return self._coerce(o) - self

    def __mul__(self, o):
        o = self._coerce(o)
        rad = abs(self.mid) * o.rad + abs(o.mid) * self.rad + self.rad * o.rad
        return _Ball._make(self.mid * o.mid, rad)

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = self._coerce(o)
        denom = abs(o.mid) - o.rad
        if not denom > 0:
            return _Ball.nan()
        mid = self.mid / o.mid
        return _Ball._make(mid, (self.rad + abs(mid) * o.rad) / denom)

    def __repr__(self):
        return f"_Ball({mp.nstr(self.mid, 15)} +- {mp.nstr(self.rad, 3)})"


ERROR_CONTROLS = ("doubling",)


def _compute_radii(z, C, d, start, length, bits):
    """Weyl radii as balls whose radii estimate the numerical error.

    Floating-point runs at ``bits`` and ``bits + guard``; the error of the more
    precise run is estimated by (twice) the difference of the two plus one ulp
    at ``bits``.  Garbage at the lower precision shows up as a large difference.
    """
    guard = max(32, bits // 4)
    status, low = _run_radii(z, C, d, start, length, bits)
    if status != "ok":
        return status, low
    status, high = _run_radii(z, C, d, start, length, bits + guard)
    if status != "ok":
        return status, high
    with _workprec(bits + guard):
        ulp = mp.ldexp(1, -bits)
        radii = []
        for row_low, row_high in zip(low, high):
            row = []
            for r1, r2 in zip(row_low, row_high):
                if not (mp.isfinite(r1) and mp.isfinite(r2) and r1 > 0 and r2 > 0):
                    row.append(_Ball.nan())
                    continue
                row.append(_Ball(r2, 2 * abs(r1 - r2) + r2 * ulp))
            radii.append(row)
    return "ok", radii


def _radii_resolved(radii, rel_tol):
    """True if every radius is a finite positive ball with relative radius <= rel_tol."""
    return all(r.is_finite() and r.lower() > 0 and r.rad <= rel_tol * r.mid
               for row in radii for r in row)


# ---------------------------------------------------------------------------
# Convergence tests
#
# All comparisons are made on balls with three-valued logic:
# True / False if the comparison holds / fails for every point in the ball,
# None if the ball straddles the threshold.  A None anywhere means the
# precision is too low to take the decision, and the run reports state -3.
# ---------------------------------------------------------------------------

def _all_lt(xs, c):
    """max(xs) < c."""
    if all(t.upper() < c for t in xs):
        return True
    if any(t.lower() >= c for t in xs):
        return False
    return None


def _all_gt(xs, c):
    """min(xs) > c."""
    if all(t.lower() > c for t in xs):
        return True
    if any(t.upper() <= c for t in xs):
        return False
    return None


def _and(*bs):
    if any(b is False for b in bs):
        return False
    if any(b is None for b in bs):
        return None
    return True


def _default_test(r, nconv, me0, me1):
    w = r[-nconv:]
    return (_all_lt([t - r[-1] for t in w], me0),
            _all_lt(w, me0),
            _all_gt(w, me1))


def _raabe_test(r, nconv, me0, me1, raabe_ratio):
    w = r[-nconv:]
    is_conv = _all_lt([t - r[-1] for t in w], me0)
    is_nonzero = _all_gt(w, me1)
    is_zero = _all_lt(w, me0)
    # Raabe criterion: (n+1)(1 - r_{n+1}/r_n)
    start = max(0, len(r) - 1 - nconv)
    raabe = [(j + 1) * (1 - r[j + 1] / r[j]) for j in range(start, len(r) - 1)]
    is_greater1 = _all_gt(raabe, raabe_ratio)
    is_smaller1 = _all_lt(raabe, 1 / raabe_ratio)
    is_zero = _and(is_zero, is_greater1)
    is_nonzero = _and(is_nonzero, is_smaller1)
    xor = None if is_zero is None or is_nonzero is None else (is_zero != is_nonzero)
    is_conv = _and(is_conv, xor)
    return is_conv, is_zero, is_nonzero


# ---------------------------------------------------------------------------
# DeficiencyIndices
# ---------------------------------------------------------------------------

def _check_error_control(error_control):
    if error_control not in ERROR_CONTROLS:
        raise ValueError(f"error_control must be one of {ERROR_CONTROLS} "
                         "(rigorous ball arithmetic needs python-flint, see block_jacobi.py)")


def _deficiency_indices_z(z, C, *, precision, start_index, algorithm_length,
                          epsilon0, epsilon1, raabe_ratio, convergence_test, simple_carleman, method):
    if not _is_hermitian(C):
        warnings.warn("Coefficient list has to be hermitian.")
        return {"state": 2, "n": 0, "zeros": 0}
    md = _max_shift(C)
    if md == 0:
        # diagonal in the number basis with real entries: self-adjoint, nothing to compute
        return {"state": 0, "n": 0, "zeros": 0}
    if simple_carleman and _max_power(C) <= 2:
        return {"state": 0, "n": 0, "zeros": md}
    bits = _digits_to_bits(precision)
    if _coefficient_precision_bits(C) < bits:
        return {"state": 1, "n": 0, "zeros": 0}

    status, radii = _compute_radii(z, C, md, start_index, algorithm_length, bits)
    if status != "ok":
        return {"state": status, "n": 0, "zeros": 0, **radii}

    with _workprec(bits):
        columns = list(zip(*radii))      # Transpose: one sequence per eigenvalue index
        # The radii entering the tests must be bounded away from 0 (Min[radia] == 0 check).
        window = convergence_test + 1
        if not all(t.is_finite() and t.lower() > 0 for col in columns for t in col[-window:]):
            return {"state": -7, "n": 0, "zeros": 0}
        # Radii are non-increasing in exact arithmetic; a certain increase signals precision loss.
        not_monotonic = any((col[k + 1] - col[k]).lower() > 0
                            for col in columns for k in range(len(col) - 1))
        if method.lower() == "raabe":
            results = [_raabe_test(list(col), convergence_test, epsilon0, epsilon1, raabe_ratio)
                       for col in columns]
        else:
            results = [_default_test(list(col), convergence_test, epsilon0, epsilon1) for col in columns]

    if any(b is None for res in results for b in res):
        return {"state": -3, "n": 0, "zeros": 0}      # decision not resolved -> precision too low
    count0 = sum(1 for res in results if res[1])
    count1 = sum(1 for res in results if res[2])
    if not_monotonic:
        state = -2
    elif not all(res[0] for res in results):
        state = -1
    elif count0 + count1 != md:
        state = -4
    else:
        state = 0
    return {"state": state, "n": count1, "zeros": count0}


def deficiency_indices(poly, z=None, *, precision=MACHINE_PRECISION, start_index=0,
                       algorithm_length=50, epsilon0=1e-3, epsilon1=1e-2, raabe_ratio=2,
                       convergence_test=25, simple_carleman=True, method="raabe",
                       error_control="doubling"):
    """Single run of the algorithm at fixed precision / length (DeficiencyIndices).

    ``poly`` is a string, a sympy expression, or a coefficient matrix.
    ``z`` defaults to computing both z = i and z = -i (returns a pair).
    ``precision`` is in decimal digits.  ``error_control`` only accepts
    "doubling" (compare runs at two precisions).  With ``method="raabe"`` a
    radius counts as zero if the Raabe quotients exceed ``raabe_ratio`` and as
    non-zero if they stay below ``1/raabe_ratio``.
    """
    _check_error_control(error_control)
    C = _as_coefficients(poly)
    opts = dict(precision=precision, start_index=start_index, algorithm_length=algorithm_length,
                epsilon0=epsilon0, epsilon1=epsilon1, raabe_ratio=raabe_ratio,
                convergence_test=convergence_test,
                simple_carleman=simple_carleman, method=method)
    if z is None:
        return (_deficiency_indices_z(1j, C, **opts), _deficiency_indices_z(-1j, C, **opts))
    return _deficiency_indices_z(z, C, **opts)


def adaptive_deficiency_indices(poly, z=None, *, epsilon0=1e-3, epsilon1=1e-2, raabe_ratio=2,
                                convergence_test=25, min_algorithm_length=50,
                                max_algorithm_length=10000, delta_algorithm_length=100,
                                min_precision=MACHINE_PRECISION, max_precision=5000,
                                min_start_index=0, simple_carleman=True, adaptive_length=True,
                                adaptive_rescaling=True, debug=False, method="raabe",
                                error_control="doubling"):
    """AdaptiveDeficiencyIndices: rerun with larger length / precision / start index until success.

    States -1, -4 increase the algorithm length; -2, -3, -5, -7, -8 double the
    precision; -6 increases the start index.  With ``z=None`` returns the pair
    (z = i, z = -i); for real coefficients n+ = n-, so only z = i is computed
    and the same dict is returned twice.
    """
    _check_error_control(error_control)
    C = _as_coefficients(poly)
    if z is None:
        kw = dict(locals())
        del kw["poly"], kw["z"], kw["C"]
        if _is_real(C):                           # real coefficients: n+ = n-, compute once
            res = adaptive_deficiency_indices(C, 1j, **kw)
            return (res, res)
        return (adaptive_deficiency_indices(C, 1j, **kw), adaptive_deficiency_indices(C, -1j, **kw))

    dom = _dominant_n(C) if adaptive_length else 1
    length = dom * min_algorithm_length
    delta = dom * delta_algorithm_length
    if adaptive_rescaling:
        C = (C * _rescaling(C)).applyfunc(sp.expand)

    state = {"state": -1, "precision": min_precision, "length": length, "start": min_start_index}
    while state["state"] < 0 and state["precision"] <= max_precision and state["length"] <= max_algorithm_length:
        res = _deficiency_indices_z(
            z, C, precision=state["precision"], start_index=state["start"],
            algorithm_length=state["length"], epsilon0=epsilon0, epsilon1=epsilon1,
            raabe_ratio=raabe_ratio, convergence_test=convergence_test, simple_carleman=simple_carleman, method=method)
        state = {**state, **res}
        if debug:
            print(state, file=sys.stderr)
        s = state["state"]
        if s in (-1, -4):
            state["length"] += delta
        elif s in (-2, -3, -5, -7, -8):
            state["precision"] *= 2
        elif s == -6:
            state["start"] += state["deltaShift"]
    return state


# ---------------------------------------------------------------------------
# DeficiencyPlot
# ---------------------------------------------------------------------------

def weyl_radii(poly, z=1j, nmax=100, *, precision=MACHINE_PRECISION, start_index=0,
               auto_scale=False, warn=True, rel_tol=1e-4, max_precision=5000,
               error_control="doubling"):
    """Weyl-circle radii r_i(N), N = 0..nmax (RadiusOperators / DeficiencyPlot).

    The precision (decimal digits) is doubled automatically until every radius
    is known to relative accuracy ``rel_tol``; singular A blocks shift the start
    index.  Returns a dict with ``radii`` (list of d sequences, descending
    eigenvalue order, like Transpose[Eigenvalues /@ ...]; floats may underflow,
    hence also ``log10_radii``), estimated ``rel_errors``, and the
    ``precision``/``start`` that were needed.
    """
    _check_error_control(error_control)
    C = _as_coefficients(poly)
    if not _is_hermitian(C) and warn:
        warnings.warn("Coefficient list has to be hermitian")
    if auto_scale:
        warnings.warn("Rescaling coefficients and Length")
        nmax = _dominant_n(C) * nmax
        C = (C * _rescaling(C)).applyfunc(sp.expand)
    elif warn and _scale_test(C):
        warnings.warn("Rescaling coefficients is recommended. Use auto_scale=True")
    d = _max_shift(C)
    if d == 0:
        raise ValueError("operator is diagonal in the number basis; there are no Weyl radii")
    coeff_bits = _coefficient_precision_bits(C)
    digits, start = precision, start_index
    while True:
        bits = _digits_to_bits(digits)
        if coeff_bits < bits:
            raise PrecisionError(
                f"coefficients only carry {coeff_bits} bits but {bits} are needed; "
                "use exact coefficients (e.g. strings or sympy Rationals)")
        status, info = _compute_radii(z, C, d, start, nmax, bits)
        if status == -6:
            start += info["deltaShift"]
            continue
        if status == "ok":
            with _workprec(bits):
                if _radii_resolved(info, rel_tol):
                    cols = list(zip(*info))
                    return {"radii": [[float(r.mid) for r in col] for col in cols],
                            "log10_radii": [[float(mp.log10(r.mid)) for r in col] for col in cols],
                            "rel_errors": [[float(r.rad / r.mid) for r in col] for col in cols],
                            "precision": digits, "start": start, "z": z}
        digits *= 2
        if digits > max_precision:
            raise PrecisionError(f"radii not resolved to rel. accuracy {rel_tol} "
                                 f"with max_precision={max_precision} digits")


def deficiency_plot(poly, z=1j, nmax=100, *, ax=None, **kwargs):
    """Plot the Weyl radii against N (log scale).  Returns (data, ax)."""
    import matplotlib.pyplot as plt

    data = weyl_radii(poly, z, nmax, **kwargs)
    if ax is None:
        _, ax = plt.subplots()
    for i, r in enumerate(data["log10_radii"]):
        ax.plot(range(len(r)), r, ".", label=f"$r_{i}$")
    ax.set_xlabel("N")
    ax.set_ylabel("log10(Weyl radius)")
    ax.set_title(f"z = {z}, precision = {data['precision']:.4g} digits")
    ax.legend()
    return data, ax


# ---------------------------------------------------------------------------
# QuickDeficiencyIndices
# ---------------------------------------------------------------------------

_COLORS = {"good": "\033[1;38;2;48;111;29m", "soso": "\033[1;38;2;226;121;46m",
           "bad": "\033[1;38;2;166;43;23m", "reset": "\033[0m"}

_ERR_GENERAL = ("Error: Failed to compute deficiency indices. Use adaptive_deficiency_indices "
                "for detailed output and deficiency_plot for visual feedback")
_ERR_NOT_CONVERGED = "Possible Problems: max_algorithm_length is not sufficient; epsilon1 is too large"
_ERR_PRECISION = "Possible Problems: max_precision is not sufficient"
_ERR_COEFFICIENTS = "Error source: Coefficients do not have sufficient precision. Increase input precision"
_ERR_HERMITIAN = "Error source: Input is not hermitian"


def _display(poly):
    s = poly if isinstance(poly, str) else sp.sstr(poly)
    return s.replace("**", "^").replace("adag", "a†")


def quick_deficiency_indices(poly, *, return_result=False, color=None, file=None, **kwargs):
    """Compute n+ and n- adaptively and print a one-line verdict.

    Returns the pair of result dicts (z = i, z = -i) on failure or if
    ``return_result`` is set.
    """
    file = file or sys.stdout
    if color is None:
        try:
            color = bool(file.isatty())
        except Exception:
            color = False
    paint = (lambda s, c: f"{_COLORS[c]}{s}{_COLORS['reset']}") if color else (lambda s, c: s)

    res = adaptive_deficiency_indices(poly, **kwargs)
    states = [r["state"] for r in res]
    header = f"H = {_display(poly)}"
    if not all(s == 0 for s in states):
        if 1 in states:
            reason = _ERR_COEFFICIENTS
        elif 2 in states:
            reason = _ERR_HERMITIAN
        elif -1 in states or -4 in states:
            reason = _ERR_NOT_CONVERGED
        else:
            reason = _ERR_PRECISION
        print(paint(header, "bad"), file=file)
        print(paint(_ERR_GENERAL, "bad"), file=file)
        print(reason, file=file)
        return res

    npl, nmi = res[0]["n"], res[1]["n"]
    if npl == 0 and nmi == 0:
        kind, label = "good", "✓ good Hamiltonian"
    elif npl == nmi:
        kind, label = "soso", "ℹ more physical information necessary"
    else:
        kind, label = "bad", "✗ bad Hamiltonian"
    print(paint(header, kind), file=file)
    print(f"n+ = {npl},   n- = {nmi}   —   {paint(label, kind)}", file=file)
    if return_result:
        return res


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} 'polynomial' ['polynomial' ...]", file=sys.stderr)
        sys.exit(1)
    for arg in sys.argv[1:]:
        quick_deficiency_indices(arg)
