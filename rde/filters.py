"""Stage 6: canonical identity, known-result matching, looseness and scoring (plan 9)."""
from __future__ import annotations

import itertools
import math
from fractions import Fraction
from typing import Dict, List, Sequence, Tuple

import sympy as sp

from .expr import VARS, expr_of
from .types import Feature, Relation

INTERIOR = Fraction(1, 1000)


def combination(coeffs: Sequence[int], features: Sequence[Feature]) -> sp.Expr:
    """Sympy expression for sum c_i f_i."""
    return sp.cancel(sum(c * expr_of(f) for c, f in zip(coeffs, features)))


def canonical_key(expr: sp.Expr) -> str:
    """Key of an expression up to positive scaling and variable permutation (plan 9, filter 1).

    Numerator and denominator are normalised separately: the denominator is made monic and the
    numerator is divided by the absolute value of its leading coefficient.
    """
    if expr == 0:
        return "0"
    best = None
    for perm in itertools.permutations(VARS):
        e = sp.cancel(expr.subs(dict(zip(VARS, perm)), simultaneous=True))
        num, den = sp.fraction(e)
        d_lc = sp.Poly(sp.expand(den), *VARS).LC()
        num, den = sp.expand(num / d_lc), sp.expand(den / d_lc)
        n_lc = sp.Poly(num, *VARS).LC()
        key = str(sp.expand(num / abs(n_lc))) + " / " + str(den)
        best = key if best is None or key < best else best
    return best


def expanded_sign_trivial(expr: sp.Expr) -> bool:
    """True if the expression is nonnegative on the positive orthant by its expanded coefficients.

    Catches combinations that are trivial only after expansion, e.g. (sum a)(sum ab) - sym[a^2 b] = 3abc.
    """
    if expr == 0:
        return False
    num, den = sp.fraction(sp.cancel(expr))
    num_c = sp.Poly(sp.expand(num), *VARS).coeffs()
    den_c = sp.Poly(sp.expand(den), *VARS).coeffs()
    if all(c >= 0 for c in den_c) and all(c >= 0 for c in num_c):
        return True
    return all(c <= 0 for c in den_c) and all(c <= 0 for c in num_c)


def parse_point(text: str) -> Tuple[Fraction, ...]:
    """Parse a grid point string such as '(1/2, 1/2, 0)'."""
    return tuple(Fraction(p) for p in text.strip("()").split(", "))


def tight_classes(rel: Relation) -> Tuple[bool, int]:
    """(has an interior tight point, number of distinct tight points up to permutation)."""
    pts = [parse_point(t) for t in rel.tight_rows]
    interior = any(min(p) > INTERIOR for p in pts)
    distinct = len({tuple(sorted(p)) for p in pts})
    return interior, distinct


def score(rel: Relation) -> float:
    """Plan 9 score: complexity penalty plus tightness bonuses."""
    complexity = sum(f.size for c, f in zip(rel.coeffs, rel.features) if c != 0)
    magnitude = sum(math.log2(1 + abs(c)) for c in rel.coeffs)
    interior, distinct = tight_classes(rel)
    return -complexity - magnitude + 4.0 * interior + 2.0 * (distinct >= 2)


def is_loose(rel: Relation) -> bool:
    """Loose inequalities have no tight grid point, so they can be strengthened (plan 8, step 5)."""
    return rel.kind == "ineq" and not rel.tight_rows


class KnownIndex:
    """Canonical keys of the benchmark's algebraic inequalities and identities (plan 9, filter 4)."""

    def __init__(self, entries: Sequence[Tuple[str, str, str]]):
        from .expr import parse
        self.keys: Dict[str, str] = {}
        for name, _kind, text in entries:
            self.keys[canonical_key(sp.cancel(parse(text)))] = name

    def match(self, expr: sp.Expr):
        """Name of the known result this expression equals up to positive scaling, or None."""
        return self.keys.get(canonical_key(expr))


def texts_combination(coeffs: Sequence[int], texts: Sequence[str]) -> sp.Expr:
    """Sympy expression for sum c_i f_i given feature texts."""
    from .expr import parse
    return sp.cancel(sum(c * parse(t) for c, t in zip(coeffs, texts)))


def implied_by(expr: sp.Expr, stored: Sequence[sp.Expr]) -> bool:
    """Plan 9, filter 3: True if expr = sum lambda_j s_j with lambda_j >= 0 for stored inequalities s_j.

    All expressions are multiplied by the common denominator, so the identity becomes a polynomial one
    in a monomial basis. The LP finds lambda; the multipliers are rationalised and the identity is
    re-checked exactly, so a float artefact cannot remove a relation.
    """
    import numpy as np
    from scipy.optimize import linprog

    if not stored or expr == 0:
        return False
    dens = [sp.fraction(sp.cancel(e))[1] for e in [expr] + list(stored)]
    L = sp.lcm_list(dens)
    polys = [sp.Poly(sp.expand(sp.cancel(e * L)), *VARS) for e in [expr] + list(stored)]
    monos = sorted({m for p in polys for m in p.monoms()})
    index = {m: i for i, m in enumerate(monos)}

    def vec(p):
        v = np.zeros(len(monos))
        for m, c in zip(p.monoms(), p.coeffs()):
            v[index[m]] = float(c)
        return v

    b = vec(polys[0])
    A = np.column_stack([vec(p) for p in polys[1:]])
    res = linprog(np.zeros(A.shape[1]), A_eq=A, b_eq=b, bounds=[(0, None)] * A.shape[1], method="highs")
    if res.status != 0:
        return False
    lam = [sp.Rational(x).limit_denominator(10 ** 6) for x in res.x]
    if any(x < 0 for x in lam):
        return False
    combo = sum(l * p.as_expr() for l, p in zip(lam, polys[1:]))
    return sp.expand(combo - polys[0].as_expr()) == 0
