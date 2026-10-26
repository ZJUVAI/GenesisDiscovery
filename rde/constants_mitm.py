"""Meet-in-the-middle screen for c0 + c1 x + c2 K + c3 x K = 0 (Ramanujan-Machine style, plan 7).

For a fixed constant x the values K = -(c0 + c1 x) / (c2 + c3 x) over a coefficient box are computed
once in float64, sorted, and each candidate K is located by binary search. Hits are candidates only;
they are re-checked at high precision by constants.relation_search-style PSLQ before they count.
"""
from __future__ import annotations

import itertools
from typing import List, Optional, Tuple

import mpmath as mp
import numpy as np

Relation = Tuple[int, int, int, int]


class Index:
    """Sorted Mobius values for one constant over the box |c_i| <= bound."""

    def __init__(self, x: float, bound: int = 12, tol: float = 1e-9):
        self.x = x
        self.tol = tol
        rng = np.arange(-bound, bound + 1, dtype=float)
        c0, c1, c2, c3 = np.meshgrid(rng, rng, rng, rng, indexing="ij")
        c0, c1, c2, c3 = c0.ravel(), c1.ravel(), c2.ravel(), c3.ravel()
        den = c2 + c3 * x
        num = c0 + c1 * x
        ok = (np.abs(den) > 1e-12) & ((c2 != 0) | (c3 != 0)) & ((c1 != 0) | (c3 != 0)) & ((c0 != 0) | (c1 != 0) | (c2 != 0) | (c3 != 0))
        vals = -num[ok] / den[ok]
        keep = np.isfinite(vals) & (np.abs(vals) < 1e6)
        vals = vals[keep]
        coeffs = np.stack([c0[ok][keep], c1[ok][keep], c2[ok][keep], c3[ok][keep]], axis=1).astype(int)
        order = np.argsort(vals)
        self.vals = vals[order]
        self.coeffs = coeffs[order]

    def lookup(self, K: float) -> List[Relation]:
        """All box relations whose Mobius value is within tol of K (candidates, not yet verified)."""
        lo = np.searchsorted(self.vals, K - self.tol)
        hi = np.searchsorted(self.vals, K + self.tol)
        return [tuple(int(c) for c in row) for row in self.coeffs[lo:hi]]


def verify(rel: Relation, x: mp.mpf, K: mp.mpf, dps: int = 60, slack_exp: int = 40) -> bool:
    """High-precision check: |c0 + c1 x + c2 K + c3 x K| below 10^-slack_exp at dps digits."""
    c0, c1, c2, c3 = rel
    with mp.workdps(dps):
        r = c0 + c1 * x + c2 * K + c3 * x * K
        return abs(r) < mp.mpf(10) ** (-slack_exp)
