"""Constants domain: continued fractions and integer relations (Ramanujan-Machine style).

A polynomial continued fraction K = a_0 + b_1/(a_1 + b_2/(a_2 + ...)) with a_n = p(n), b_n = q(n).
The search asks PSLQ for integers (c0, c1, c2, c3) with c0 + c1*x + c2*K + c3*x*K = 0, i.e.
x = -(c0 + c2*K) / (c1 + c3*K). Every hit is re-checked at higher precision before it is reported.
"""
from __future__ import annotations

import math
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import mpmath as mp

Poly = Tuple[int, ...]  # coefficients in increasing powers of n

CONSTANTS = {
    "pi": lambda: mp.pi,
    "e": lambda: mp.e,
    "log2": lambda: mp.log(2),
    "zeta3": lambda: mp.zeta(3),
    "catalan": lambda: mp.catalan,
    "zeta5": lambda: mp.zeta(5),
}


def _poly_eval(coeffs: Poly, n: int) -> int:
    """Evaluate an integer polynomial, coefficients in increasing powers."""
    return sum(c * n ** i for i, c in enumerate(coeffs))


def pcf_value(p: Poly, q: Poly, depth: int) -> mp.mpf:
    """K = p(0) + q(1)/(p(1) + q(2)/(p(2) + ...)) truncated at the given depth, by backward recurrence."""
    tail = mp.mpf(_poly_eval(p, depth))
    for n in range(depth - 1, -1, -1):
        tail = _poly_eval(p, n) + _poly_eval(q, n + 1) / tail
    return tail


def pcf_converged(p: Poly, q: Poly, depth: int, dps: int) -> Optional[mp.mpf]:
    """Value at two depths; accept only if they agree to dps/2 digits and no term vanishes."""
    with mp.workdps(dps):
        if any(_poly_eval(q, n) == 0 for n in range(1, depth + 1)):
            return None
        try:
            a = pcf_value(p, q, depth)
            b = pcf_value(p, q, 2 * depth)
        except ZeroDivisionError:
            return None
        if not (mp.isfinite(a) and mp.isfinite(b)) or abs(a) > mp.mpf(10) ** 6:
            return None
        if abs(a - b) > mp.mpf(10) ** (-(dps // 2)) * max(1, abs(a)):
            return None
        return b


def polynomial_pairs(max_coeff: int, deg_p: int = 1, deg_q: int = 2) -> Iterator[Tuple[Poly, Poly]]:
    """All (p, q) with p of degree <= deg_p, q of degree <= deg_q and coefficients in [-max_coeff, max_coeff].

    Pairs with q identically zero or p(n) = 0 for some n in 0..3 are skipped.
    """
    rng = range(-max_coeff, max_coeff + 1)
    for p in _all_polys(deg_p, rng):
        if any(_poly_eval(p, n) == 0 for n in range(4)):
            continue
        for q in _all_polys(deg_q, rng):
            if any(q):
                yield p, q


def _all_polys(deg: int, rng: range) -> Iterator[Poly]:
    """Every coefficient tuple of length deg+1 with entries in rng."""
    if deg == 0:
        for c in rng:
            yield (c,)
        return
    for head in _all_polys(deg - 1, rng):
        for c in rng:
            yield head + (c,)


def relation_search(K: mp.mpf, x: mp.mpf, dps: int = 60, maxcoeff: int = 500) -> Optional[Tuple[int, int, int, int]]:
    """PSLQ for (c0, c1, c2, c3) with c0 + c1 x + c2 K + c3 x K = 0, nontrivial in both x and K."""
    with mp.workdps(dps):
        vec = [mp.mpf(1), x, K, x * K]
        rel = mp.pslq(vec, maxcoeff=maxcoeff, maxsteps=20000)
    if rel is None:
        return None
    c0, c1, c2, c3 = rel
    # nontrivial: the relation must involve K and x (otherwise it is a relation among 1 and x, or 1 and K)
    if (c2, c3) == (0, 0) or (c1, c3) == (0, 0):
        return None
    return tuple(int(r) for r in rel)


def is_rational(K: mp.mpf, dps: int = 60) -> bool:
    """True if PSLQ finds a small rational relation for K (then K is not a useful feature)."""
    with mp.workdps(dps):
        if abs(K) < mp.mpf(10) ** (-dps // 2):
            return True
        try:
            return mp.pslq([mp.mpf(1), K], maxcoeff=10 ** 6, maxsteps=10000) is not None
        except ValueError:
            return True


def verify_relation(rel: Sequence[int], p: Poly, q: Poly, name: str, dps: int = 180) -> Optional[float]:
    """Re-evaluate K and x at higher precision; return log10 of |residual| (very negative is good)."""
    with mp.workdps(dps):
        K = pcf_converged(p, q, 600, dps)
        if K is None:
            return None
        x = CONSTANTS[name]()
        resid = rel[0] + rel[1] * x + rel[2] * K + rel[3] * x * K
        return float(mp.log10(abs(resid) + mp.mpf(10) ** (-dps)))


def describe_relation(rel: Sequence[int], name: str) -> str:
    """Closed form x = -(c0 + c2 K)/(c1 + c3 K) written as text."""
    c0, c1, c2, c3 = rel
    num = _lin(c0, c2, "K")
    den = _lin(c1, c3, "K")
    return f"{name} = -({num})/({den})"


def _lin(a: int, b: int, sym: str) -> str:
    """Text for a + b*sym."""
    parts = []
    if a:
        parts.append(str(a))
    if b:
        coef = "" if abs(b) == 1 else f"{abs(b)}*"
        sign = "-" if b < 0 else ("+" if parts else "")
        parts.append(f"{sign}{coef}{sym}" if parts else f"{'-' if b < 0 else ''}{coef}{sym}")
    return " ".join(parts) if parts else "0"


def check_known_4_over_pi(dps: int = 60) -> Tuple[bool, str]:
    """Plan M3 acceptance: a_n = 2n+1, b_n = n^2 gives K with pi*K = 4 (K = 4/pi)."""
    with mp.workdps(dps):
        K = pcf_converged((1, 2), (0, 0, 1), 400, dps)
        if K is None:
            return False, "did not converge"
        val = mp.pi * K
        ok = abs(val - 4) < mp.mpf(10) ** (-(dps - 10))
        return ok, f"pi*K = {mp.nstr(val, 25)}"


def search(names: Sequence[str], max_coeff: int, dps_screen: int = 60, depth: int = 400,
           progress=None) -> List[Dict]:
    """Scan polynomial continued fractions with small coefficients against each named constant."""
    hits: List[Dict] = []
    seen_values = set()
    mp.mp.dps = dps_screen + 40
    count = 0
    for p, q in polynomial_pairs(max_coeff):
        count += 1
        K = pcf_converged(p, q, depth, dps_screen + 40)
        if K is None:
            continue
        key = mp.nstr(K, 40)
        if key in seen_values or is_rational(K):
            continue
        seen_values.add(key)
        for name in names:
            x = CONSTANTS[name]()
            rel = relation_search(K, x, dps=dps_screen, maxcoeff=max_coeff)
            if rel is None:
                continue
            log_res = verify_relation(rel, p, q, name)
            if log_res is None or log_res > -(120):
                continue
            hits.append({"constant": name, "p": p, "q": q, "relation": rel,
                         "closed_form": describe_relation(rel, name), "log10_residual": round(log_res, 1)})
        if progress and count % 5000 == 0:
            progress(count, len(seen_values), len(hits))
    return hits
