"""Vectorised float64 screen for polynomial continued fractions (plan 5 / Ramanujan-style search).

K = p(0) + q(1)/(p(1) + q(2)/(p(2) + ...)) for many (p, q) at once, by backward recurrence over numpy
arrays. Float64 values are candidates only; survivors are re-checked at high precision elsewhere.
"""
from __future__ import annotations

import numpy as np


def cf_float(a: np.ndarray, b: np.ndarray, depth: int) -> np.ndarray:
    """K for each row: a is (N, deg_p + 1) coefficients of p, b is (N, deg_q + 1) coefficients of q.

    Uses increasing-power coefficients, as in rde.constants.
    """
    def poly(coef: np.ndarray, n: float) -> np.ndarray:
        out = np.zeros(coef.shape[0])
        for i in range(coef.shape[1] - 1, -1, -1):
            out = out * n + coef[:, i]
        return out

    with np.errstate(all="ignore"):
        t = poly(a, float(depth))
        for n in range(depth - 1, -1, -1):
            t = poly(a, float(n)) + poly(b, float(n + 1)) / t
    return t
