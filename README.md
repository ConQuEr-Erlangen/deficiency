This Mathematica package determines the deficiency indices (n₊, n₋) of Hermitian polynomials in the position and momentum operators x and p, or equivalently in the creation and annihilation operators a† and a, of a single bosonic mode. The deficiency indices decide whether such an operator defines exactly one unitary dynamics (n₊ = n₋ = 0), a family of physically distinct dynamics (n₊ = n₋ ≠ 0), or no unitary dynamics at all (n₊ ≠ n₋). The input is the matrix of coefficients of the polynomial. The repository contains deficiency.wl, which provides the relevant functions, and exampes.nb, where the algorithm is used to determine deficiency indices of operators found in the literature. The most important functions for the user are:

* QuickDeficiencyIndices[poly_]: Returns the deficiency indices of poly_
* AdaptiveDeficiencyIndices[poly_]: The implementation of QuickDeficiencyIndices, displaying all the metadata and error codes
* DeficiencyPlot[poly_,z_,n_]: Returns the first n radii, ready to be plotted with list plot

A web interface to the algorithm is available at https://conquer-erlangen.github.io/deficiency/
A python version can be found in the docs directory.
We stress that the Mathematica package is the tested reference implemention.
