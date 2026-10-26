"""M6: each known result is checked numerically on its own (plan 12). Failures mean a wrong fixture."""
import random

import mpmath as mp
import numpy as np
import pytest
import sympy as sp

from rde.domains import Positive3
from rde.expr import VARS, parse
from tests.known_results import ALGEBRA, CONSTANTS

DOMAIN = Positive3()
_RNG = random.Random(2026)
_POINTS = [tuple(_RNG.uniform(1e-3, 1.0) for _ in range(3)) for _ in range(3000)]
_POINTS = [tuple(x / sum(p) for x in p) for p in _POINTS] + DOMAIN.special_points()


def _values(text):
    fn = sp.lambdify(VARS, parse(text), "mpmath")
    out = []
    with mp.workdps(50):
        for p in _POINTS:
            try:
                out.append(fn(*[mp.mpf(x) for x in p]))
            except ZeroDivisionError:
                continue
    return out


@pytest.mark.parametrize("name, kind, text", ALGEBRA, ids=[a[0] for a in ALGEBRA])
def test_algebraic_known_result(name, kind, text):
    vals = _values(text)
    assert len(vals) > 2900
    if kind == "ineq":
        assert min(vals) >= -mp.mpf(10) ** -30, f"{name}: min {min(vals)}"
    else:
        assert max(abs(v) for v in vals) < mp.mpf(10) ** -30, f"{name}: not identically zero"


@pytest.mark.parametrize("name, residual", CONSTANTS, ids=[c[0] for c in CONSTANTS])
def test_constant_known_result(name, residual):
    with mp.workdps(60):
        r = residual(60)
        assert abs(r) < mp.mpf(10) ** -40, f"{name}: residual {r}"


def test_benchmark_size():
    assert len(ALGEBRA) >= 25 and len(CONSTANTS) >= 15


def test_benchmark_entries_are_pairwise_distinct():
    # two names for one inequality (up to positive scale and variable permutation) would corrupt recall
    from rde.filters import canonical_key
    from rde.expr import parse
    import sympy as sp
    keys = {}
    for name, kind, text in ALGEBRA:
        if kind != "ineq":
            continue
        key = canonical_key(sp.cancel(parse(text)))
        assert key not in keys, f"{name} duplicates {keys.get(key)}"
        keys[key] = name
