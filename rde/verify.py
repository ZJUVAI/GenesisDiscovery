"""Stage 7 level 1 (plan 10): fresh-sample numeric test at high precision, with optimiser restarts."""
from __future__ import annotations

from typing import Sequence, Tuple

import mpmath as mp
import numpy as np
import sympy as sp

from .domains import Positive3
from .expr import VARS, expr_of
from .types import Feature

FRESH_ROWS = 5000  # plan 10
RESTARTS = 20      # plan 10
DPS = 100          # plan 10: 2x the working precision of 50
TOL = mp.mpf(10) ** -30


def level1(coeffs: Sequence[int], features: Sequence[Feature], rng: np.random.Generator) -> Tuple[str, str]:
    """('refuted' | 'numeric', min slack as a string). A violation at any fresh point refutes the relation."""
    exprs = [expr_of(f) for f in features]
    domain = Positive3()
    rows = domain.sample(FRESH_ROWS, rng)
    fns = [sp.lambdify(VARS, e, "mpmath") for e in exprs]
    worst = None
    with mp.workdps(DPS):
        for r in rows:
            x = [mp.mpf(v) for v in r]
            try:
                vals = [fn(*x) for fn in fns]
            except (ZeroDivisionError, ValueError):
                continue
            num = sum(c * v for c, v in zip(coeffs, vals))
            den = sum(abs(c * v) for c, v in zip(coeffs, vals))
            if den == 0:
                continue
            slack = num / den
            if worst is None or slack < worst:
                worst = slack
        if worst is not None and worst < -TOL:
            return "refuted", mp.nstr(worst, 6)

    def neg_slack(p):
        x = [mp.mpf(v) for v in p]
        try:
            with mp.workdps(DPS):
                vals = [fn(*x) for fn in fns]
                return float(sum(c * v for c, v in zip(coeffs, vals)) / sum(abs(c * v) for c, v in zip(coeffs, vals)))
        except (ZeroDivisionError, ValueError):
            return float("inf")

    for _ in range(RESTARTS):
        pt, val = domain.minimize(neg_slack, 1, rng)
        if val < -1e-12:
            with mp.workdps(DPS):
                x = [mp.mpf(v) for v in pt]
                vals = [fn(*x) for fn in fns]
                slack = sum(c * v for c, v in zip(coeffs, vals)) / sum(abs(c * v) for c, v in zip(coeffs, vals))
            if slack < -TOL:
                return "refuted", mp.nstr(slack, 6)
    return "numeric", "" if worst is None else mp.nstr(worst, 6)
