"""Exact finite checks supporting, not replacing, the PL-F+ proofs.

Standard library only. Enumerates all nonempty 3-by-3 occupancy masks,
including holes, point contacts, and globally reconnecting local sectors.
Checks sectorwise nodal field assignments, exposed-edge normals, shared
edge continuity, and the elementary square-spectrum ceiling inequality.
No eigenvalue computation and no numerical claim of a general PDE proof.
"""

if not __debug__:
    raise RuntimeError("Bookkeeping checks require assertions: run Python without -O or -OO.")

from itertools import product
import json
from math import isqrt


def sector_value(occupied, vertex, cell):
    x, y = vertex
    incident = {(x - 1, y - 1), (x, y - 1), (x - 1, y), (x, y)}
    incident &= occupied
    sector, pending = {cell}, [cell]
    while pending:
        a, b = pending.pop()
        for d in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)):
            if d in incident and d not in sector:
                sector.add(d)
                pending.append(d)
    normals = set()
    for a, b in sector:
        nx = -1 if x == a else 1
        ny = -1 if y == b else 1
        if (a + nx, b) not in occupied:
            normals.add((nx, 0))
        if (a, b + ny) not in occupied:
            normals.add((0, ny))
    h = tuple(sum(n[i] for n in normals) for i in (0, 1))
    assert all(abs(v) <= 1 for v in h)
    assert sum(v * v for v in h) <= 2
    return h


def check_mask(occupied):
    values = {}
    for a, b in occupied:
        for v in product((a, a + 1), (b, b + 1)):
            values[(a, b), v] = sector_value(occupied, v, (a, b))
    for a, b in occupied:
        edges = [
            ((-1, 0), ((a, b), (a, b + 1))),
            ((1, 0), ((a + 1, b), (a + 1, b + 1))),
            ((0, -1), ((a, b), (a + 1, b))),
            ((0, 1), ((a, b + 1), (a + 1, b + 1))),
        ]
        for normal, endpoints in edges:
            neighbor = a + normal[0], b + normal[1]
            for vertex in endpoints:
                h = values[(a, b), vertex]
                if neighbor in occupied:
                    assert h == values[neighbor, vertex]
                else:
                    assert sum(h[i] * normal[i] for i in (0, 1)) == 1
        for i in (0, 1):
            for u, v in (((a, b), (a + 1, b)),
                         ((a, b + 1), (a + 1, b + 1)),
                         ((a, b), (a, b + 1)),
                         ((a + 1, b), (a + 1, b + 1))):
                assert abs(values[(a, b), u][i] - values[(a, b), v][i]) <= 2


cells = list(product(range(3), repeat=2))
for bits in range(1, 1 << len(cells)):
    check_mask({c for i, c in enumerate(cells) if bits & (1 << i)})

for m in range(1, 100001):
    a = isqrt(m)
    a += a * a < m
    b = (m + a - 1) // a
    assert a * b >= m
    assert a * a + b * b <= 3 * m

for M in range(12, 10001):
    count = sum(m % 6 in (1, 5) for m in range(M, 2 * M + 1))
    assert 6 * count >= M

print(json.dumps({
    "status": "PASS",
    "occupancy_masks_checked": 511,
    "spectrum_ceiling_integer_cases": 100000,
    "sharpness_residue_intervals_checked": 9989,
    "arithmetic": "exact integers",
    "scope": "finite bookkeeping checks; not a substitute for the analytical proofs"
}, indent=2))

