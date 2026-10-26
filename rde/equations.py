"""Stage 4 for the constants domain: PSLQ integer relations among named constants (plan 7)."""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Tuple

import mpmath as mp

RECHECK_MARGIN = 30  # digits: the 2D check must reach 10^-(2D - RECHECK_MARGIN)


def required_digits(k: int, max_coeff: int = 1000) -> int:
    """Working precision D = max(100, ceil(2 k log10 max_coeff) + 40) for k features (plan 7)."""
    return max(100, math.ceil(2 * k * math.log10(max_coeff)) + 40)


def pslq_relation(evaluate: Callable[[], Sequence[mp.mpf]], names: Sequence[str], target: str,
                  max_coeff: int = 1000) -> Optional[Tuple[int, ...]]:
    """Integer relation c with sum c_i v_i = 0 among the named values, or None.

    `evaluate` recomputes the values at the current mpmath precision. A relation found at D digits
    must hold to 10^-(2D - 30) when the values are recomputed at 2D digits.
    """
    if target not in names:
        raise ValueError(f"target {target!r} is not among the features")
    k = len(names)
    D = required_digits(k, max_coeff)
    with mp.workdps(D):
        rel = mp.pslq(list(evaluate()), maxcoeff=max_coeff, maxsteps=200000)
    if rel is None or sum(1 for c in rel if c) < 2:
        return None
    if rel[list(names).index(target)] == 0:
        return None
    with mp.workdps(2 * D):
        residual = abs(sum(c * v for c, v in zip(rel, evaluate())))
    if residual >= mp.mpf(10) ** (-(2 * D - RECHECK_MARGIN)):
        return None
    first = next(c for c in rel if c)
    if first < 0:
        rel = [-c for c in rel]
    return tuple(int(c) for c in rel)


def find_constant_relations(evaluate: Callable[[], Sequence[mp.mpf]], names: Sequence[str], target: str,
                            max_coeff: int = 1000) -> list:
    """Relation for the target, then (plan 7 step 5) a rerun after dropping the largest feature found."""
    found = []
    live = list(names)
    while target in live and len(live) >= 2:
        def sub_eval(live=tuple(live)):
            full = list(evaluate())
            return [full[names.index(n)] for n in live]
        rel = pslq_relation(sub_eval, live, target, max_coeff)
        if rel is None:
            break
        found.append((tuple(live), rel))
        used = [n for n, c in zip(live, rel) if c and n != target]
        if not used:
            break
        live = [n for n in live if n not in used[:1]]
    return found
