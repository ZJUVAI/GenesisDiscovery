"""Algebraic domain (positive variables on the simplex a+b+c=1): table, equations, inequalities, proofs.

Rows are a rational grid on the simplex. Float64 is used only to locate candidate vertices of the
sampled cone; every accepted relation is recomputed exactly over Fractions and sympy, and inequalities
are proved by the ordered-difference method (plan section 10, level 2).
"""
from __future__ import annotations

import itertools
import math
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

import mpmath as mp
import numpy as np
import sympy as sp
from scipy.optimize import linprog

from .expr import VARS, expr_of
from .types import Feature, Relation

DPS = 50
MAX_COEFF = 200  # plan 6: coefficient bound for rationalisation
FLOAT_TOL = 1e-7  # plan 6: tolerance for coefficients from a float64 solver
GRID_N = 30
X, U, V = sp.symbols("x u v")
Terms = Dict[Tuple[int, int, int], Fraction]


BOUNDARY_EPS = (Fraction(1, 1000), Fraction(1, 10 ** 6), Fraction(1, 10 ** 9))  # plan 5


def boundary_points() -> List[Tuple[Fraction, Fraction, Fraction]]:
    """Plan 5 near-boundary and near-corner points, exact: they reach limits such as a << b << c."""
    pts = set()
    for eps in BOUNDARY_EPS:
        for t in (Fraction(1, 10), Fraction(3, 10), Fraction(1, 2)):
            base = (t, 1 - t - eps, eps)
            pts.update(itertools.permutations(base))
        pts.update(itertools.permutations((1 - 2 * eps, eps, eps)))
        # hierarchical corner a << b << c (not in plan 5): reaches infima such as inf cyc[a/(a+b)] = 1
        pts.update(itertools.permutations((eps * eps, eps, 1 - eps - eps * eps)))
    return sorted(pts)


def grid_points(n: int = GRID_N, boundary: bool = True) -> List[Tuple[Fraction, Fraction, Fraction]]:
    """Rational points (i, j, k)/n on the simplex, excluding the three corners, plus boundary points."""
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            if sorted((i, j, k))[:2] == [0, 0]:
                continue
            pts.append((Fraction(i, n), Fraction(j, n), Fraction(k, n)))
    if boundary:
        pts = sorted(set(pts) | set(boundary_points()))
    return pts


def fmt_point(p: Sequence[Fraction]) -> str:
    """Readable text for a rational point."""
    return "(" + ", ".join(str(x) for x in p) + ")"


def evaluate_table(exprs: Sequence[sp.Expr], pts) -> Tuple[list, np.ndarray]:
    """Evaluate every expression on every point at DPS digits; drop rows where any value fails."""
    mp.mp.dps = DPS
    fns = [sp.lambdify(VARS, e, "mpmath") for e in exprs]
    kept, rows = [], []
    for p in pts:
        x = [mp.mpf(q.numerator) / q.denominator for q in p]
        try:
            vals = [fn(*x) for fn in fns]
        except (ZeroDivisionError, ValueError, TypeError):
            continue
        if any((not mp.isfinite(v)) or abs(v) > mp.mpf(10) ** 30 for v in vals):
            continue
        kept.append(p)
        rows.append([float(v) for v in vals])
    return kept, np.array(rows, dtype=float)


def _poly_terms(e: sp.Expr) -> Terms:
    """Monomial coefficients of a polynomial as exact Fractions."""
    poly = sp.Poly(sp.expand(e), *VARS)
    return {m: Fraction(int(sp.numer(c)), int(sp.denom(c))) for m, c in zip(poly.monoms(), poly.coeffs())}


def _eval_terms(terms: Terms, p) -> Fraction:
    """Exact evaluation of a polynomial given by its monomial dictionary at a rational point."""
    total = Fraction(0)
    for (i, j, k), c in terms.items():
        total += c * p[0] ** i * p[1] ** j * p[2] ** k
    return total


def exact_matrix(exprs: Sequence[sp.Expr], pts) -> List[List[Fraction]]:
    """Exact feature values: one row per grid point, one column per feature."""
    parts = []
    for e in exprs:
        num, den = sp.fraction(sp.together(e))
        parts.append((_poly_terms(num), _poly_terms(den)))
    return [[_eval_terms(nt, p) / _eval_terms(dt, p) for nt, dt in parts] for p in pts]


def to_integer_vector(vec: Sequence[float], max_den: int = 60, max_coeff: int = 200,
                      tol: float = 1e-7, first_positive: bool = True) -> Optional[Tuple[int, ...]]:
    """Rationalise a float vector to a primitive integer vector, or return None."""
    arr = [float(x) for x in vec]
    if not any(arr):
        return None
    pivot = max(range(len(arr)), key=lambda i: abs(arr[i]))
    arr = [x / arr[pivot] for x in arr]
    fracs = []
    for x in arr:
        fr = Fraction(x).limit_denominator(max_den)
        if abs(float(fr) - x) > tol * max(1.0, abs(x)):
            return None
        fracs.append(fr)
    lcm = 1
    for fr in fracs:
        lcm = lcm * fr.denominator // math.gcd(lcm, fr.denominator)
    ints = [int(fr * lcm) for fr in fracs]
    g = 0
    for z in ints:
        g = math.gcd(g, abs(z))
    ints = [z // g for z in ints]
    if first_positive:
        first = next(z for z in ints if z != 0)
        if first < 0:
            ints = [-z for z in ints]
    if max(abs(z) for z in ints) > max_coeff:
        return None
    return tuple(ints)


def _combination(coeffs: Sequence[int], exprs: Sequence[sp.Expr]) -> sp.Expr:
    """Exact rational function sum c_i f_i."""
    return sp.cancel(sp.Add(*[c * e for c, e in zip(coeffs, exprs)]))


def find_equations(features: Sequence[Feature], exprs: Sequence[sp.Expr], M: np.ndarray):
    """Greedy span test (plan 7, many-rows mode). Returns (equations, independent indices)."""
    basis: List[int] = []
    equations: List[Relation] = []
    for idx in range(len(features)):
        if not basis:
            basis.append(idx)
            continue
        T = M[:, basis]
        y = M[:, idx]
        w, *_ = np.linalg.lstsq(T, y, rcond=None)
        resid = np.linalg.norm(T @ w - y) / max(np.linalg.norm(y), 1e-300)
        if resid < 1e-9:
            cols = basis + [idx]
            ints = to_integer_vector(list(w) + [-1.0], first_positive=True)
            if ints is not None and _combination(ints, [exprs[c] for c in cols]) == 0:
                equations.append(Relation("eq", ints, tuple(features[c] for c in cols), status="proved"))
                continue
        basis.append(idx)
    return equations, basis


def _extreme_rays(T: np.ndarray, n_objectives: int, rng: np.random.Generator) -> List[np.ndarray]:
    """Vertices of {c : Tc >= 0, 1'Tc = 1} for random linear objectives (plan 8, step 1)."""
    n, k = T.shape
    A_eq = T.sum(axis=0).reshape(1, k)
    rays: List[np.ndarray] = []
    seen = set()
    for _ in range(n_objectives):
        u = rng.exponential(size=n)
        res = linprog(T.T @ u, A_ub=-T, b_ub=np.zeros(n), A_eq=A_eq, b_eq=[1.0],
                      bounds=[(None, None)] * k, method="highs-ds")
        if res.status != 0:
            continue
        key = tuple(np.round(res.x, 7))
        if key not in seen:
            seen.add(key)
            rays.append(res.x)
    return rays


def exact_ray(T_float: np.ndarray, T_exact: List[List[Fraction]], ray: np.ndarray) -> Optional[Tuple[int, ...]]:
    """Exact primitive integer ray at a float LP vertex, or None if it is not a vertex on the grid."""
    row_scale = np.abs(T_float).sum(axis=1) * max(np.abs(ray).sum(), 1e-300)
    residual = np.abs(T_float @ ray)
    tight = [i for i in range(len(T_exact)) if residual[i] <= 1e-9 * max(row_scale[i], 1e-300)]
    if not tight:
        return None
    null = sp.Matrix([[sp.Rational(x.numerator, x.denominator) for x in T_exact[i]] for i in tight]).nullspace()
    if len(null) != 1:
        return None
    v = null[0]
    den = sp.ilcm(*[sp.fraction(x)[1] for x in v])
    v = [int(x * den) for x in v]
    g = 0
    for z in v:
        g = math.gcd(g, abs(z))
    v = [z // g for z in v]
    if max(abs(z) for z in v) > MAX_COEFF:
        return None
    values = [sum(a * b for a, b in zip(row, v)) for row in T_exact]
    if all(x >= 0 for x in values):
        return tuple(v)
    if all(x <= 0 for x in values):
        return tuple(-z for z in v)
    return None


def _nonneg_coefficients(e: sp.Expr) -> Optional[int]:
    """+1 if numerator and denominator have all coefficients >= 0, -1 if all <= 0, else None."""
    num, den = sp.fraction(sp.cancel(e))
    signs = set()
    for part in (num, den):
        coeffs = sp.Poly(sp.expand(part), *VARS).coeffs()
        if all(c >= 0 for c in coeffs):
            signs.add(1)
        elif all(c <= 0 for c in coeffs):
            signs.add(-1)
        else:
            return None
    return signs.pop() if len(signs) == 1 else None


def is_sign_trivial(coeffs: Sequence[int], exprs: Sequence[sp.Expr]) -> bool:
    """True if every term c_i f_i is nonnegative on the simplex by coefficient inspection."""
    for c, e in zip(coeffs, exprs):
        if c == 0:
            continue
        s = _nonneg_coefficients(e)
        if s is None or s * (1 if c > 0 else -1) < 0:
            return False
    return True


def ordered_difference_proof(P: sp.Expr, max_lift: int = 2) -> Optional[str]:
    """Prove a homogeneous polynomial P >= 0 on the orthant by the ordered-difference substitution.

    For each ordering p <= q <= r of (a, b, c), substitute p = x, q = x + u, r = x + u + v.
    If every coefficient of every expansion is nonnegative, P >= 0. Multiplying by (a+b+c)^m
    is tried for m = 0..max_lift; the multiplier is positive, so the direction is kept.
    """
    for m in range(max_lift + 1):
        Q = sp.expand(P * sp.Add(*VARS) ** m)
        ok = True
        for perm in itertools.permutations(VARS):
            sub = {perm[0]: X, perm[1]: X + U, perm[2]: X + U + V}
            expansion = sp.Poly(sp.expand(Q.subs(sub, simultaneous=True)), X, U, V)
            if any(c < 0 for c in expansion.coeffs()):
                ok = False
                break
        if ok:
            return f"ordered-difference, multiplier (a+b+c)^{m}"
    return None


def exact_proof(coeffs: Sequence[int], exprs: Sequence[sp.Expr]) -> Optional[str]:
    """Attempt a symbolic proof that sum c_i f_i >= 0 on the simplex. Returns the method or None."""
    P = _combination(coeffs, exprs)
    if P == 0:
        return "identity"
    num, den = sp.fraction(sp.together(P))
    den_coeffs = sp.Poly(sp.expand(den), *VARS).coeffs()
    if all(c >= 0 for c in den_coeffs):
        sign = 1
    elif all(c <= 0 for c in den_coeffs):
        sign = -1
    else:
        return None
    return ordered_difference_proof(sp.expand(sign * num))


def _slack_at(coeffs: Sequence[int], fns, xy: np.ndarray) -> Tuple[float, float]:
    """Normalised slack at chart point (a, b) with c = 1 - a - b; returns (slack, c)."""
    a, b = float(xy[0]), float(xy[1])
    c = 1.0 - a - b
    pt = [np.array([a]), np.array([b]), np.array([c])]
    terms = np.array([float(k) * float(np.asarray(fn(*pt)).reshape(-1)[0]) for k, fn in zip(coeffs, fns)])
    scale = np.abs(terms).sum()
    with np.errstate(all="ignore"):
        value = float(terms.sum() / scale) if scale > 0 else 1.0
    return (value if np.isfinite(value) else 1.0), c


def numeric_min_slack(coeffs: Sequence[int], exprs: Sequence[sp.Expr], rng: np.random.Generator,
                      samples: int = 4000, polish: int = 8) -> Tuple[float, Tuple[float, float, float]]:
    """Smallest normalised slack over the simplex, with the worst point.

    Random simplex samples, then Nelder-Mead from the best few. The slack is
    sum c_i f_i / sum |c_i f_i|, so it lies in [-1, 1] and is scale-free.
    """
    from scipy.optimize import minimize

    fns = [sp.lambdify(VARS, e, "numpy") for e in exprs]
    x = rng.dirichlet((1.0, 1.0, 1.0), size=samples)
    vals = [_slack_at(coeffs, fns, x[i, :2])[0] for i in range(samples)]
    best = (min(vals), x[int(np.argmin(vals))])
    for start in np.array(x[np.argsort(vals)[:polish], :2]):
        res = minimize(lambda xy: _slack_at(coeffs, fns, xy)[0], start, method="Nelder-Mead",
                       options={"xatol": 1e-10, "fatol": 1e-14, "maxfev": 400})
        if res.fun < best[0] and res.x[0] >= 0 and res.x[1] >= 0 and res.x[0] + res.x[1] <= 1:
            best = (float(res.fun), np.array([res.x[0], res.x[1], 1 - res.x[0] - res.x[1]]))
    return best[0], tuple(float(t) for t in best[1])


def exact_values(coeffs: Sequence[int], T_exact: List[List[Fraction]]) -> List[Fraction]:
    """Exact value of sum c_i f_i at each grid row."""
    return [sum(c * x for c, x in zip(coeffs, row)) for row in T_exact]


def tight_grid_points(coeffs: Sequence[int], T_exact: List[List[Fraction]], pts) -> List[str]:
    """Grid points where the combination is exactly zero."""
    return [fmt_point(p) for p, v in zip(pts, exact_values(coeffs, T_exact)) if v == 0]


def _float_row(exprs: Sequence[sp.Expr], point: Tuple[float, float, float]) -> List[float]:
    """Feature values at a float simplex point, at DPS digits, as floats."""
    mp.mp.dps = DPS
    x = [mp.mpf(v) for v in point]
    return [float(sp.lambdify(VARS, e, "mpmath")(*x)) for e in exprs]


def inequalities(features: Sequence[Feature], exprs: Sequence[sp.Expr], basis: List[int],
                 M: np.ndarray, pts, rng: np.random.Generator, n_objectives: int = 200,
                 max_rounds: int = 8) -> List[Relation]:
    """Extreme rays of the sampled cone (plan 8), with the adversarial loop (step 4).

    Each round solves the LP on the current rows, recovers candidate integer vectors, checks them
    exactly on the grid, then minimises the normalised slack over the simplex. A violating point
    becomes a new row. Rounds stop when no row is added or after max_rounds.
    """
    sub_features = tuple(features[i] for i in basis)
    sub_exprs = [exprs[i] for i in basis]
    T_float = M[:, basis]
    T_exact = exact_matrix(sub_exprs, pts)
    results: List[Relation] = []
    settled = set()
    for _ in range(max_rounds):
        added = 0
        for ray in _extreme_rays(T_float, n_objectives, rng):
            ints = exact_ray(T_float, T_exact, ray) or to_integer_vector(ray, max_den=2000, max_coeff=MAX_COEFF, tol=FLOAT_TOL)
            if ints is None or ints in settled:
                continue
            values = exact_values(ints, T_exact)
            if any(x < 0 for x in values):
                continue
            slack, worst = numeric_min_slack(ints, sub_exprs, rng)
            if slack < -1e-9:
                T_float = np.vstack([T_float, np.array(_float_row(sub_exprs, worst))])
                added += 1
                continue
            settled.add(ints)
            if is_sign_trivial(ints, sub_exprs):
                continue
            method = exact_proof(ints, sub_exprs)
            status = "proved" if method is not None else "numeric"
            results.append(Relation(
                kind="ineq", coeffs=ints, features=sub_features,
                tight_rows=tuple(tight_grid_points(ints, T_exact, pts)),
                min_slack=f"{slack:.3e}", status=status, score=0.0,
            ))
        if added == 0:
            break
    return results


def target_bounds(features: Sequence[Feature], exprs: Sequence[sp.Expr], basis: List[int],
                  M: np.ndarray, pts) -> List[Relation]:
    """Second mode (plan 8): for each feature f, the largest mean lower bound f >= sum_j w_j g_j.

    Solves max sum_i w.g_i subject to f_i >= w.g_i on every row, then rationalises and checks
    each bound exactly on the grid. The target coefficient is kept positive.
    """
    sub_features = tuple(features[i] for i in basis)
    sub_exprs = [exprs[i] for i in basis]
    T_exact = exact_matrix(sub_exprs, pts)
    M = M[:, basis]
    results: List[Relation] = []
    for t in range(len(basis)):
        others = [j for j in range(len(basis)) if j != t]
        if not others:
            continue
        G = M[:, others]
        res = linprog(-G.sum(axis=0), A_ub=G, b_ub=M[:, t], bounds=[(None, None)] * len(others),
                      method="highs-ds")
        if res.status != 0:
            continue
        vec = np.zeros(len(basis))
        vec[t] = 1.0
        vec[others] = -res.x
        ints = to_integer_vector(vec, first_positive=False)
        if ints is None:
            continue
        if ints[t] < 0:
            ints = tuple(-z for z in ints)
        values = exact_values(ints, T_exact)
        if any(x < 0 for x in values):
            continue
        if is_sign_trivial(ints, sub_exprs):
            continue
        method = exact_proof(ints, sub_exprs)
        results.append(Relation(
            kind="ineq", coeffs=ints, features=sub_features,
            tight_rows=tuple(tight_grid_points(ints, T_exact, pts)),
            min_slack="0", status="proved" if method is not None else "numeric", score=0.0,
        ))
    return results
