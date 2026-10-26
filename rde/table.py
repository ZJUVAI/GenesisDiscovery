"""Stage 3: evaluate features on rows and clean the table (plan 6)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import mpmath as mp
import numpy as np
import sympy as sp

from .domains import Positive3, default_rows
from .expr import VARS, expr_of
from .types import Feature, Task

MAX_ABS = mp.mpf(10) ** 30
MAX_FAIL_SHARE = 0.05
MIN_ROWS_PER_FEATURE = 5


class TooFewRows(RuntimeError):
    """Raised when cleaning leaves fewer than 5 rows per feature (plan 6 step 6)."""


@dataclass
class Table:
    """Cleaned table: exact-precision values, float copy for LP work, and the cleaning log."""

    rows: List[Tuple[float, ...]]
    features: List[Feature]
    values: List[List[mp.mpf]]
    M: np.ndarray
    scale: np.ndarray
    dps: int
    duplicates: List[Tuple[str, str]] = field(default_factory=list)
    dropped_features: List[str] = field(default_factory=list)
    dropped_rows: int = 0


def _evaluate(exprs: Sequence[sp.Expr], rows: Sequence[Tuple[float, ...]], dps: int):
    """Values at dps digits, with None where a feature fails (pole, complex, non-finite)."""
    mp.mp.dps = dps
    fns = [sp.lambdify(VARS, e, "mpmath") for e in exprs]
    grid = []
    for r in rows:
        x = [mp.mpf(v) for v in r]
        out = []
        for fn in fns:
            try:
                v = fn(*x)
                out.append(v if (mp.isfinite(v) and abs(v) <= MAX_ABS) else None)
            except (ZeroDivisionError, ValueError, TypeError):
                out.append(None)
        grid.append(out)
    return grid


def build_table(task: Task, domain: Positive3, rng: np.random.Generator, dps: int = 50,
                rows: Optional[Sequence[Tuple[float, ...]]] = None) -> Table:
    """Build and clean the table for a task (plan 6, steps 1-6)."""
    exprs = [expr_of(f) for f in task.features]
    features = list(task.features)
    rows = list(rows) if rows is not None else default_rows(domain, len(features), rng)
    grid = _evaluate(exprs, rows, dps)
    dropped: List[str] = []
    # step 3: drop features that failed on too many rows (before row removal)
    keep_cols = []
    for j, f in enumerate(features):
        fails = sum(1 for row in grid if row[j] is None)
        if fails > MAX_FAIL_SHARE * len(grid):
            dropped.append(f.text)
        else:
            keep_cols.append(j)
    # step 2: drop rows with any failed kept feature
    keep_rows = [i for i, row in enumerate(grid) if all(row[j] is not None for j in keep_cols)]
    n_dropped_rows = len(grid) - len(keep_rows)
    values = [[grid[i][j] for j in keep_cols] for i in keep_rows]
    features = [features[j] for j in keep_cols]
    kept_rows = [rows[i] for i in keep_rows]
    # step 4: duplicate columns agreeing to relative 10^-(dps-10)
    tol = mp.mpf(10) ** (-(dps - 10))
    duplicates: List[Tuple[str, str]] = []
    drop_cols = set()
    for j in range(len(features)):
        for i in range(j):
            if i in drop_cols:
                continue
            if all(abs(row[i] - row[j]) <= tol * max(1, abs(row[i])) for row in values):
                duplicates.append((features[i].text, features[j].text))
                drop_cols.add(j)
                break
    if drop_cols:
        keep = [j for j in range(len(features)) if j not in drop_cols]
        values = [[row[j] for j in keep] for row in values]
        features = [features[j] for j in keep]
    if len(values) < MIN_ROWS_PER_FEATURE * max(1, len(features)):
        raise TooFewRows(f"{len(values)} rows for {len(features)} features")
    M = np.array([[float(v) for v in row] for row in values], dtype=float)
    # step 5: column scales (median of absolute values)
    scale = np.median(np.abs(M), axis=0)
    scale[scale == 0] = 1.0
    return Table(rows=kept_rows, features=features, values=values, M=M / scale, scale=scale, dps=dps,
                 duplicates=duplicates, dropped_features=dropped, dropped_rows=n_dropped_rows)
