# Sharp torsion stability and certified spectral surgery

**Author:** R. Arun Chandru  
**Date:** 2 October 2026



## Extended accessible abstract

The frequencies of an ideal drum depend on its shape. The mathematical
object describing those frequencies is the Dirichlet spectrum. A different
experiment applies a constant load to the same domain and measures its
equilibrium response. The integral of this response is its torsional
rigidity. This paper asks how much information about the entire spectrum
is controlled by a change in that single quantity, and how the resulting
estimates can support finite, certified optimization over shapes.

For a domain contained in another domain, the loss of torsional rigidity
controls a precise aggregate difference between their inverse Laplacians.
This applies to arbitrary finite-volume open sets in every dimension at
least two, retaining the actual Dirichlet boundary condition even on
measure-zero sets of positive capacity. The paper determines the best
possible powers of torsion loss in the relevant Schatten norms. These
norms measure all singular values of an operator together, not just its
largest one. Matching examples explain the change of exponent: one small
removed component and many small removed components impose different
limitations. The examples preserve the outer volume and lowest
eigenvalue.

The paper also characterizes exactly which continuous scalar functions
of the reciprocal eigenvalues have a uniform linear torsion bound.
Heat traces and regularized Fredholm determinants are applications.
For nonnested domains, the proved comparison uses the torsion cost of
passing through their intersection. This cost is also the least total
torsion change along a finite chain of nested domain changes.

A complementary estimate is stronger when only a finite energy window
is considered: both the norm and the trace of a compressed resolvent
difference depend linearly on torsion loss. In the plane, this permits
surgery that either preserves an initial spectral list or exchanges a
controlled loss of spectral rank for removed physical area. Smoothing,
covering, localization, and component packing then lead to explicit
finite libraries of polygonal domains. These approximate a prescribed
optimal eigenvalue, a common class of monotone finite-spectrum
objectives, and the infimum of area times eigenvalue divided by index.
The bounds account separately for the number of cells, boundary edges,
indices, and candidate geometries.

To certify the continuum spectra of those geometries, the paper clips
the spectrum of a surrounding square's Laplacian at height H and
compresses it to the chosen domain. A finite matrix determines the
resulting lower spectral levels. The continuum error is bounded by an
explicit multiple of H^(-1/2), uniformly over all finite unit-square
masks, including holes, disconnected pieces, and point contacts.
One fixed rectangular geometry proves that the decay exponent cannot
be improved. A further argument covers masks from supplied
shape-regular triangular meshes, with explicit geometric conditioning.
Subtracting the clipped inverse's constant plateau also provides
finite-rank approximation of the whole shifted resolvent.


## Main quantitative statements

- In dimension n, put d=n/2 and r=d+1. For nested domains A inside D,
  with physical torsion loss delta=T(D)-T(A),
  the r-th Schatten power of the zero-extended inverse difference
  and the ordered r-th inverse-power trace difference are bounded
  by c_n delta, where c_n=2^(2-d)/(pi^d Gamma(d+1)).
- For p>d under a volume bound, the optimal uniform torsion exponent
  in the Schatten norm is min(1-d/p,1/r). The exponent is sharp;
  the numerical upper constants are not asserted optimal.
- On a unit-cell mask, every eigenvalue at most E has clipping error
  at most 256(E+1)^(3/2)/sqrt(H). A supplied triangular mesh has an
  analogous bound involving its minimum angle and altitude.
- Fixed-index optimization has a sufficient cell budget
  ceil(125000000 k^2 epsilon^(-2)); the all-index infimum has budgets
  ceil(2^46 epsilon^(-6)) cells and ceil(302 epsilon^(-3)) indices.
  Boundary encoding improves the logarithmic enumeration bounds.
- A normalized prefix of length m on a mask of at most B cells has
  a sufficient correction-matrix dimension O(B^4 m^3 tau^(-2)).
  The arithmetic appendix proves polynomial per-mask bit cost in
  these numerical parameters.


## Contents of this upload folder

| File | Purpose |
|---|---|
| r_arun_chandru_Sharp torsion stability and certified spectral surgery.pdf | Full paper, including figures, references, and appendix |
| manuscript.tex | Single self-contained LaTeX source; no external bibliography or figure files |
| verify_spectra.py | Exact-arithmetic verification of four small spectral examples |
| certificates.json | Recorded output of the spectral verifier |
| check_geometry.py | Finite local mask and integer-bookkeeping checks |
| README.md | This overview and reproduction instructions |

## Compile the manuscript

Use a reasonably recent TeX distribution with AMS-LaTeX, Latin Modern,
geometry, microtype, TikZ, xurl, and hyperref.
All figures and the bibliography are embedded in the source.

Run from this folder:

~~~sh
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
~~~

Alternatively:

~~~sh
tectonic manuscript.tex
~~~

## Reproduce the finite checks

The spectral verifier requires Python 3.10 or newer and NumPy.
The geometry checker uses the Python standard library.
Do not run either script with Python's optimization flags:
the verification assertions must remain enabled.

~~~sh
python verify_spectra.py --output certificates_reproduced.json
python check_geometry.py
~~~

The full spectral run checks six indices on each of four geometries:
a unit square, a three-cell L-shaped domain, a rational rectangle
with nearby distinct levels, and two squares whose closures touch
at one point. Multiple clipping heights and conforming meshes are
used. Floating-point eigensolvers propose vectors; rational
orthogonality, residual, interval, and Rayleigh--Ritz checks decide
whether a certificate is accepted. The output records the exact
enclosures, parameters, and verifier-source hash.
Small floating-point proposal differences across NumPy/BLAS versions
may change outward endpoints while preserving the certified tests.
The script limits its numerical linear algebra to one thread.

The local geometry check examines every nonempty occupancy pattern in
a 3-by-3 cell block, the associated sector-field compatibility, and
finite integer ranges used to test the mode-count bookkeeping.
These finite checks complement the all-size arguments in the paper;
they do not replace them.