"""Stage 2: domains, row sampling and violating-point search (plan 5)."""
from __future__ import annotations

import itertools
from typing import Callable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import differential_evolution, minimize

Point = Tuple[float, float, float]
BOUNDARY_T = (0.1, 0.3, 0.5)
BOUNDARY_EPS = (1e-3, 1e-6, 1e-9)
CORNER_EPS = (1e-3, 1e-6, 1e-9, 1e-12)
TWO_EQUAL_T = (0.01, 0.1, 0.2, 0.25, 0.4, 0.45, 0.49)
SIMPLEX_TOL = 1e-12


def _perms(p: Point) -> List[Point]:
    """All distinct coordinate permutations of a point."""
    return sorted({tuple(q) for q in itertools.permutations(p)})


def _chart(p: Sequence[float]) -> Point:
    """Map the unit square onto the open simplex: a = u, b = (1-u)v, c = (1-u)(1-v)."""
    u, v = p
    return (u, (1 - u) * v, (1 - u) * (1 - v))


class Positive3:
    """Positive a, b, c with a + b + c = 1 (valid because each task has one homogeneous degree)."""

    name = "positive3"
    variables: Tuple[str, ...] = ("a", "b", "c")

    def sample(self, n: int, rng: np.random.Generator) -> List[Point]:
        """Rows in the plan's mix: 60% Dirichlet(1), 20% Dirichlet(0.2), 20% two-equal points."""
        n1 = round(0.6 * n)
        n2 = round(0.2 * n)
        n3 = n - n1 - n2
        rows: List[Point] = [tuple(x) for x in rng.dirichlet((1, 1, 1), size=n1)]
        rows += [tuple(x) for x in rng.dirichlet((0.2, 0.2, 0.2), size=n2)]
        for _ in range(n3):
            t = rng.uniform(0, 0.5)
            rows.append(tuple(rng.permutation([t, t, 1 - 2 * t])))
        return [tuple(float(x) for x in r) for r in rows]

    def special_points(self) -> List[Point]:
        """Centre, two-equal, near-boundary and near-corner points (all permutations)."""
        pts = {(1 / 3, 1 / 3, 1 / 3)}
        for t in TWO_EQUAL_T:
            pts.update(_perms((t, t, 1 - 2 * t)))
        for t in BOUNDARY_T:
            for eps in BOUNDARY_EPS:
                pts.update(_perms((t, 1 - t - eps, eps)))
        for eps in CORNER_EPS:
            pts.update(_perms((1 - 2 * eps, eps, eps)))
        return sorted(pts)

    def contains(self, x: Sequence[float]) -> bool:
        """True if x is strictly positive and on the simplex."""
        return min(x) > 0 and abs(sum(x) - 1) < SIMPLEX_TOL

    def minimize(self, fn: Callable[[Point], float], restarts: int, rng: np.random.Generator) -> Tuple[Point, float]:
        """Global search on the simplex: differential evolution per restart, Nelder-Mead polish of the best."""
        def objective(p):
            """Chart value, or +inf outside the unit square (Nelder-Mead is unbounded and can leave it)."""
            if not (0 < p[0] < 1 and 0 < p[1] < 1):
                return np.inf
            return fn(_chart(p))

        best_x, best_v = None, np.inf
        for _ in range(restarts):
            seed = int(rng.integers(2 ** 31))
            res = differential_evolution(objective, [(1e-9, 1 - 1e-9)] * 2, seed=seed, tol=1e-12)
            if res.fun < best_v:
                best_x, best_v = res.x, res.fun
        polished = minimize(objective, best_x, method="Nelder-Mead", options={"xatol": 1e-12, "fatol": 1e-15})
        if polished.fun < best_v:
            best_x, best_v = polished.x, polished.fun
        return _chart(best_x), float(best_v)


def default_rows(domain: Positive3, k: int, rng: np.random.Generator) -> List[Point]:
    """Plan 5: max(200, 20 k) sampled rows plus the special points."""
    return domain.sample(max(200, 20 * k), rng) + domain.special_points()
