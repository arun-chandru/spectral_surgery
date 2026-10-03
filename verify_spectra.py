"""Independent, indexed PL-F+ Dirichlet certificates for rational rectangular masks.

Only NumPy is needed for UNTRUSTED floating-point basis proposals. All matrix
assembly, interval endpoints, and certificate checks use Python exact integers
or Fraction. Run with --output PATH to save the reproducible JSON results.
This is a bounded research prototype, not a catalogue enumeration or a formal
verification of Python/NumPy/the hardware. See the accompanying manuscript and README.md.
"""
from __future__ import annotations

if not __debug__:
    raise RuntimeError("Certificate checks require assertions: run Python without -O or -OO.")

import os
for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
from math import factorial, isqrt, lcm
from pathlib import Path
import time
import numpy as np

BITS = 88
S = 1 << BITS
VBITS = 48
V = 1 << VBITS
DBITS = 42
D = 1 << DBITS


def ceildiv(a: int, b: int) -> int:
    assert b > 0
    return -((-a) // b)


class I:
    """Closed outward-rounded dyadic interval [lo/S, hi/S]."""
    __slots__ = ("lo", "hi")

    def __init__(self, lo: int, hi: int):
        assert lo <= hi
        self.lo, self.hi = lo, hi

    @staticmethod
    def rat(x):
        x = F(x)
        return I((x.numerator * S) // x.denominator,
                 ceildiv(x.numerator * S, x.denominator))

    def __add__(self, other):
        other = other if isinstance(other, I) else I.rat(other)
        return I(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __neg__(self):
        return I(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -(other if isinstance(other, I) else I.rat(other))

    def __rsub__(self, other):
        return I.rat(other) - self

    def __mul__(self, other):
        other = other if isinstance(other, I) else I.rat(other)
        z = [self.lo * other.lo, self.lo * other.hi,
             self.hi * other.lo, self.hi * other.hi]
        return I(min(z) // S, ceildiv(max(z), S))

    __rmul__ = __mul__

    def inv(self):
        assert not self.lo <= 0 <= self.hi
        # x -> 1/x is decreasing on either side of zero.
        a, b = F(S * S, self.hi), F(S * S, self.lo)
        return I(a.numerator // a.denominator,
                 ceildiv(b.numerator, b.denominator))

    def __truediv__(self, other):
        other = other if isinstance(other, I) else I.rat(other)
        return self * other.inv()

    def sqrt(self):
        assert self.lo >= 0
        a, b = isqrt(self.lo * S), isqrt(self.hi * S)
        return I(a, b + int(b * b < self.hi * S))


def atan_bounds(q: int, terms=90):
    # The alternating series has decreasing positive term sizes.
    val = sum((F((-1) ** n, (2*n + 1) * q ** (2*n + 1))
               for n in range(terms)), F(0))
    nxt = F((-1) ** terms, (2*terms + 1) * q ** (2*terms + 1))
    return min(val, val+nxt), max(val, val+nxt)


def pi_interval():
    a, b = atan_bounds(5)
    c, d = atan_bounds(239)
    lo, hi = 16*a - 4*d, 16*b - 4*c
    return I(I.rat(lo).lo, I.rat(hi).hi)


PI = pi_interval()


@lru_cache(maxsize=None)
def sinpi(r: F):
    r = F(r) % 2
    if r > 1:
        r -= 2
    if r > F(1, 2):
        r = 1-r
    elif r < F(-1, 2):
        r = -1-r
    if not r:
        return I.rat(0)
    x = PI * r
    xx = x*x
    term, ans = x, x
    # Taylor polynomial through degree 51. Lagrange remainder bounded by
    # |x|^52/52!: the next derivative is bounded by one for real x.
    for n in range(1, 26):
        term = -(term * xx) / ((2*n)*(2*n+1))
        ans = ans + term
    xabs = F(max(abs(x.lo), abs(x.hi)), S)
    rem = xabs ** 52 / factorial(52)
    ri = ceildiv(rem.numerator * S, rem.denominator)
    return I(ans.lo-ri, ans.hi+ri)


@lru_cache(maxsize=None)
def one_dim_gram(p: int, q: int, a: F, b: F):
    """2 integral_a^b sin(p*pi*t)sin(q*pi*t) dt, rational a,b."""
    if a == 0 and b == 1:
        return I.rat(int(p == q))
    def primitive_difference(k):
        return (sinpi(k*b)-sinpi(k*a))/(PI*k)
    if p == q:
        return I.rat(b-a) - primitive_difference(2*p)
    return primitive_difference(p-q)-primitive_difference(p+q)


def row_norm(a):
    return max((sum(abs(x) for x in row) for row in a), default=0)


def sqrt_fraction_upper(n: int, den: int):
    r = isqrt(n)
    return F(r + int(r*r < n), den)


def full_symmetric_certificate(a, scale):
    """All ORDERED eigenvalues of exact symmetric a/scale, no gap assumption.

    Floating values are only witnesses. Exact full-basis invertibility,
    residual, Gram defect, and min-max comparisons certify their errors.
    """
    n = len(a)
    af = np.array(a, dtype=float)/float(scale)
    vals, vecs = np.linalg.eigh(af)
    ds = [int(round(float(x)*D)) for x in vals]
    assert ds == sorted(ds)
    q = [[int(round(float(vecs[i,j])*V)) for j in range(n)] for i in range(n)]
    qt = list(zip(*q))
    gram = [[sum(x*y for x,y in zip(qt[i],qt[j]))
             - (V*V if i == j else 0) for j in range(n)] for i in range(n)]
    g = F(row_norm(gram), V*V)
    assert g < F(1, 2), "Proposed basis is not certified invertible"
    residual = [[D*sum(x*y for x,y in zip(a[i],qt[j]))
                 - scale*q[i][j]*ds[j] for j in range(n)] for i in range(n)]
    rn, cn = row_norm(residual), row_norm(zip(*residual))
    r = sqrt_fraction_upper(rn*cn, scale*V*D)
    dmax = F(max(abs(d) for d in ds), D)
    err = ((1+g)*r + 2*g*dmax)/(1-g)
    return [F(d,D) for d in ds], err, {"dimension": n,
        "gram_defect": record(g), "residual_norm_upper": record(r),
        "ordered_eigenvalue_error": record(err)}


def clipping(case, cutoff: int, count: int):
    t0 = time.perf_counter()
    b = F(case["box_side"])
    rects = case_rectangles(case)
    assert all(0 <= x0 < x1 <= b and 0 <= y0 < y1 <= b for x0,x1,y0,y1 in rects)
    modes = [(p,q) for p in range(1,isqrt(cutoff)+1)
             for q in range(1,isqrt(cutoff)+1) if p*p+q*q < cutoff]
    n = len(modes)
    a = [[0]*n for _ in range(n)]
    errors = [[0]*n for _ in range(n)]
    for i,(p,q) in enumerate(modes):
        wi = cutoff-p*p-q*q
        for j in range(i+1):
            r,s = modes[j]
            wj = cutoff-r*r-s*s
            entry = I.rat(0)
            for x0,x1,y0,y1 in rects:
                entry += (one_dim_gram(p,r,x0/b,x1/b)
                          * one_dim_gram(q,s,y0/b,y1/b))
            entry *= I.rat(wi*wj).sqrt()
            mid = (entry.lo+entry.hi)//2
            radius = max(mid-entry.lo, entry.hi-mid)
            a[i][j] = a[j][i] = mid
            errors[i][j] = errors[j][i] = radius
    matrix_err = F(row_norm(errors),S)
    vals, finite_err, cert = full_symmetric_certificate(a,S)
    total_mu_err = matrix_err+finite_err
    factor = PI*PI/(b*b)
    bounds = []
    for j in range(1,count+1):
        mu_upper = vals[-j]+total_mu_err if j <= n else F(0)
        assert mu_upper >= 0
        gap = F(cutoff)-mu_upper
        # Nonnegative Dirichlet spectrum permits clamping a negative lower bound.
        level = factor*gap
        lower = max(F(0),F(level.lo,S))
        bounds.append(record(lower))
    # Both errors, including pi, are composed in the actual final endpoint.
    err_energy = F(factor.hi,S)*total_mu_err
    cutoff_err = F(cutoff*(factor.hi-factor.lo),S)
    # Adding E to an approximation already within E gives a possible 2E
    # endpoint overhang. Include pi-factor uncertainty and final conversions.
    total_endpoint_gap = 2*err_energy+cutoff_err+F(factor.hi,S*S)+F(1,S)
    cert.update({"cutoff_integer":cutoff, "entry_interval_bits":BITS,
        "matrix_operator_error":record(matrix_err),
        "energy_matrix_and_solver_error_upper":record(err_energy),
        "energy_cutoff_factor_width":record(cutoff_err),
        "total_lower_endpoint_gap_upper":record(total_endpoint_gap),
        "cutoff_pi_scaled":True,
        "lower_bounds":bounds, "seconds":round(time.perf_counter()-t0,4)})
    return cert


def case_rectangles(case):
    sx,sy = F(case.get("sx","1")), F(case.get("sy","1"))
    return [(sx*x,sx*(x+1),sy*y,sy*(y+1)) for x,y in case["cells"]]


def fem_matrices(case, subdivisions):
    n = subdivisions
    sx,sy = F(case.get("sx","1")), F(case.get("sy","1"))
    fine = {(n*x+i,n*y+j) for x,y in case["cells"]
            for i in range(n) for j in range(n)}
    vertices = {(x+i,y+j) for x,y in fine for i,j in ((0,0),(1,0),(1,1),(0,1))}
    # A vertex lies in the physical interior exactly when all four incident
    # small squares belong to the mask. Point-pinch nodes are thus excluded.
    interior = sorted(v for v in vertices if all((v[0]+i,v[1]+j) in fine
                         for i,j in ((-1,-1),(-1,0),(0,-1),(0,0))))
    idx = {v:i for i,v in enumerate(interior)}
    size = len(interior)
    k = [dict() for _ in interior]
    m = [dict() for _ in interior]
    for x,y in sorted(fine):
        for tri in (((x,y),(x+1,y),(x+1,y+1)),((x,y),(x+1,y+1),(x,y+1))):
            bb = [tri[(i+1)%3][1]-tri[(i+2)%3][1] for i in range(3)]
            cc = [tri[(i+2)%3][0]-tri[(i+1)%3][0] for i in range(3)]
            for i,vi in enumerate(tri):
                if vi not in idx:
                    continue
                a = idx[vi]
                for j,vj in enumerate(tri):
                    if vj not in idx:
                        continue
                    b = idx[vj]
                    ke = ((sy/sx)*bb[i]*bb[j]+(sx/sy)*cc[i]*cc[j])/2
                    me = sx*sy*F(2 if i == j else 1,24*n*n)
                    k[a][b] = k[a].get(b,F(0))+ke
                    m[a][b] = m[a].get(b,F(0))+me
    kd = lcm(*(v.denominator for row in k for v in row.values()))
    md = lcm(*(v.denominator for row in m for v in row.values()))
    ki = [{j:int(v*kd) for j,v in row.items()} for row in k]
    mi = [{j:int(v*md) for j,v in row.items()} for row in m]
    return ki,kd,mi,md,interior


def dense_float(rows,den):
    a = np.zeros((len(rows),len(rows)))
    for i,row in enumerate(rows):
        for j,v in row.items():
            a[i,j] = v/den
    return a


def fem_certificate(case, subdivisions, count, basis_columns=None):
    t0 = time.perf_counter()
    k,kd,m,md,nodes = fem_matrices(case,subdivisions)
    size = len(nodes)
    assert size >= count
    cols = size if basis_columns is None else min(size,basis_columns)
    assert cols >= count
    kf,mf = dense_float(k,kd),dense_float(m,md)
    ch = np.linalg.cholesky(mf)
    temp = np.linalg.solve(ch,kf)
    transformed = np.linalg.solve(ch,temp.T).T
    vals,y = np.linalg.eigh((transformed+transformed.T)/2)
    cf = np.linalg.solve(ch.T,y[:,:cols])
    # A complete basis certifies the full discrete spectrum. An explicitly
    # recorded smaller basis still gives valid indexed continuum upper bounds.
    ds = [int(round(float(x)*D)) for x in vals[:cols]]
    assert ds == sorted(ds)
    c = [[int(round(float(cf[i,j])*V)) for j in range(cols)] for i in range(size)]
    ct = list(zip(*c))
    mc = [[sum(v*c[z][j] for z,v in m[i].items()) for j in range(cols)]
          for i in range(size)]
    kc = [[sum(v*c[z][j] for z,v in k[i].items()) for j in range(cols)]
          for i in range(size)]
    mct,kct = list(zip(*mc)),list(zip(*kc))
    gd,hd = md*V*V,kd*V*V*D
    gnum,hnum = [],[]
    for i in range(cols):
        gr,hr = [],[]
        for j in range(cols):
            gr.append(sum(x*z for x,z in zip(ct[i],mct[j]))-(gd if i == j else 0))
            hr.append(D*sum(x*z for x,z in zip(ct[i],kct[j]))
                      -(kd*V*V*ds[i] if i == j else 0))
        gnum.append(gr)
        hnum.append(hr)
    g,e = F(row_norm(gnum),gd),F(row_norm(hnum),hd)
    assert g < F(1,2), "Exact physical-mass Gram test failed"
    uppers, lowers = [],[]
    for d in ds[:count]:
        d = F(d,D)
        assert d-e > 0
        lowers.append(record((d-e)/(1+g)))
        uppers.append(record((d+e)/(1-g)))
    return {"subdivisions_per_cell":subdivisions, "dimension":size,
        "trial_dimension":cols,
        "stiffness_denominator":kd,"mass_denominator":md,
        "physical_mass_gram_defect":record(g),"energy_gram_error":record(e),
        "complete_basis":cols == size,"trial_ritz_lower_bounds":lowers,
        "continuum_upper_bounds":uppers,
        "total_trial_ritz_endpoint_gap_upper":record(max(F(hi["exact"])-F(lo["exact"])
            for hi,lo in zip(uppers,lowers))),
        "seconds":round(time.perf_counter()-t0,4)}


def decimal_outward(x: F, places=10, upper=False):
    scale = 10**places
    q = ceildiv(x.numerator*scale,x.denominator) if upper else x.numerator*scale//x.denominator
    sign = "-" if q < 0 else ""
    q = abs(q)
    return sign+str(q//scale)+"."+str(q%scale).zfill(places)


def record(x: F):
    x = F(x)
    return {"exact":str(x),"decimal_lower":decimal_outward(x),
            "decimal_upper":decimal_outward(x,upper=True),"approx":float(x)}


def analytic_rectangles(case, count):
    # Reference spectrum only for a single rectangular component; ordered by
    # exact rational weights, so no float-ordering or lost multiplicity.
    if len(case["cells"]) != 1:
        return None
    sx,sy = F(case.get("sx","1")),F(case.get("sy","1"))
    modes = sorted((F(p*p)/(sx*sx)+F(q*q)/(sy*sy),p,q)
                   for p in range(1,count+2) for q in range(1,count+2))
    # Any omitted p or q >= count+2 has energy above count explicit
    # competitors (1,q) or (p,1), respectively, for these near-square cases.
    cutoff_floor = min(F((count+2)**2)/(sx*sx)+1/(sy*sy),
                       1/(sx*sx)+F((count+2)**2)/(sy*sy))
    assert modes[count-1][0] < cutoff_floor
    refs=[]
    for w,p,q in modes[:count]:
        iv = PI*PI*w
        refs.append({"mode":[p,q],"weight":str(w),
                     "lower":record(F(iv.lo,S)),"upper":record(F(iv.hi,S))})
    return refs


CASES = [
    {"name":"unit_square", "cells":[[0,0]], "box_side":"1",
     "cutoffs":[16,64], "meshes":[4,8],"refined_trial_mesh":16},
    {"name":"three_cell_L", "cells":[[0,0],[1,0],[0,1]], "box_side":"2",
     "cutoffs":[64,144,256,512], "meshes":[4,8],"refined_trial_mesh":16},
    {"name":"near_square_rectangle", "cells":[[0,0]], "box_side":"1001/1000",
     "sx":"1", "sy":"1001/1000", "cutoffs":[64,144], "meshes":[4,8],
     "refined_trial_mesh":16},
    {"name":"point_touching_two_squares", "cells":[[0,0],[1,1]], "box_side":"2",
     "cutoffs":[64,144], "meshes":[4,8],"refined_trial_mesh":16},
]


def self_checks():
    assert PI.lo > 3*S and PI.hi < ceildiv(22*S,7)
    assert (I.rat(F(1,3))*3).lo <= S <= (I.rat(F(1,3))*3).hi
    assert sinpi(F(1,2)).lo <= S <= sinpi(F(1,2)).hi
    assert sinpi(F(-1,2)).lo <= -S <= sinpi(F(-1,2)).hi
    assert sinpi(F(1)).lo == sinpi(F(1)).hi == 0
    for p in range(1,5):
        for q in range(1,5):
            iv = one_dim_gram(p,q,F(0),F(1))
            assert iv.lo == iv.hi == S*int(p == q)
    # Exact repeated finite-matrix level, checked through the same solver.
    values,err,_ = full_symmetric_certificate([[2,0,0],[0,5,0],[0,0,5]],1)
    assert all(abs(v-e) <= err for v,e in zip(values,[2,5,5]))
    values,err,_ = full_symmetric_certificate([[3,2,3],[2,6,6],[3,6,11]],1)
    assert all(abs(v-e) <= err for v,e in zip(values,[2,2,16]))
    ki,kd,mi,md,_ = fem_matrices(CASES[0],2)
    assert len(ki) == 1 and F(ki[0][0],kd) == 4 and F(mi[0][0],md) == F(1,8)


def run(count=6, names=None):
    self_checks()
    answer = {"schema":"PL-F+ exact-arithmetic certificates v1",
              "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "arithmetic":"Exact Python integers/Fraction; outward dyadic elementary intervals",
              "numpy_role":"Untrusted basis proposals only", "numpy_version":np.__version__,
              "interval_bits":BITS,"vector_bits":VBITS,"eigenvalue_bits":DBITS,
              "pi_interval":{"lo":str(F(PI.lo,S)),"hi":str(F(PI.hi,S))},
              "cases":[]}
    for case in CASES:
        if names and case["name"] not in names:
            continue
        sx,sy = F(case.get("sx","1")),F(case.get("sy","1"))
        assert sx > 0 and sy > 0
        assert len(set(map(tuple,case["cells"]))) == len(case["cells"])
        item = {"domain":case,"physical_area":str(len(case["cells"])*sx*sy),
                "index_count":count,"lower_stages":[],"upper_stages":[]}
        for cutoff in case["cutoffs"]:
            print(f"{case['name']}: clipping cutoff {cutoff}",flush=True)
            item["lower_stages"].append(clipping(case,cutoff,count))
        for mesh in case["meshes"]:
            print(f"{case['name']}: exact conforming FEM mesh {mesh}",flush=True)
            item["upper_stages"].append(fem_certificate(case,mesh,count))
        if "refined_trial_mesh" in case:
            mesh = case["refined_trial_mesh"]
            print(f"{case['name']}: exact rational trial subspace on mesh {mesh}",flush=True)
            item["upper_stages"].append(fem_certificate(case,mesh,count,basis_columns=count+2))
        refs=analytic_rectangles(case,count)
        if case["name"] == "point_touching_two_squares":
            unit = analytic_rectangles(CASES[0],count)
            refs = [ref for ref in unit for _ in range(2)][:count]
        item["analytic_reference"] = refs
        enclosures=[]
        for j in range(count):
            lo=max(F(s["lower_bounds"][j]["exact"]) for s in item["lower_stages"])
            hi=min(F(s["continuum_upper_bounds"][j]["exact"]) for s in item["upper_stages"])
            assert lo <= hi
            if refs:
                assert lo <= F(refs[j]["upper"]["exact"])
                assert hi >= F(refs[j]["lower"]["exact"])
            enclosures.append({"index":j+1,"lower":record(lo),"upper":record(hi),
                "area_lambda_over_index_lower":record(len(case["cells"])*sx*sy*lo/(j+1)),
                "area_lambda_over_index_upper":record(len(case["cells"])*sx*sy*hi/(j+1))})
        item["combined_indexed_enclosures"]=enclosures
        answer["cases"].append(item)
    return answer


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path)
    parser.add_argument("--case",action="append")
    parser.add_argument("--count",type=int,default=6)
    args=parser.parse_args()
    result=run(args.count,args.case)
    if args.output:
        args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    for item in result["cases"]:
        print(item["domain"]["name"],"area",item["physical_area"])
        for e in item["combined_indexed_enclosures"]:
            print(e["index"],e["lower"]["decimal_lower"],e["upper"]["decimal_upper"])

