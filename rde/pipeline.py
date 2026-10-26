"""Stages 1-8 for one task at a time (plan 2). Workers are pure; the parent does global dedupe in order."""
from __future__ import annotations

import itertools
from typing import Dict, List, Sequence

import numpy as np
import sympy as sp

from . import algebra as al
from .expr import canonical, enumerate_features, expr_of
from .filters import KnownIndex, canonical_key, combination, expanded_sign_trivial, is_loose, score
from .types import Feature, Task
from .verify import level1

MIN_POOL = 4  # plan 4: a degree needs at least 4 features to be proposed
RULE_MIN_ROWS = 5  # plan 6 step 6


def build_pool(max_degree: int, extra: Sequence[str]) -> Dict[int, List[Feature]]:
    """Enumerated symmetric features plus configured extra features, grouped by degree."""
    pool = {d: list(fs) for d, fs in enumerate_features("positive3", max_degree).items()}
    seen = {f.text for fs in pool.values() for f in fs}
    for text in extra:
        feat, _ = canonical(text)
        if feat.text not in seen and feat.degree is not None:
            pool.setdefault(feat.degree, []).append(feat)
            seen.add(feat.text)
    for d in pool:
        pool[d].sort(key=lambda f: (f.size, f.text))
    return pool


def baseline_tasks(pool: Dict[int, List[Feature]], n: int, rng: np.random.Generator,
                   k_min: int = 4, k_max: int = 12) -> List[Task]:
    """Plan 4 baseline: uniform degree, uniform k, features drawn with probability proportional to 1/size."""
    degrees = [d for d in sorted(pool) if len(pool[d]) >= MIN_POOL]
    tasks: List[Task] = []
    for _ in range(n):
        d = degrees[int(rng.integers(len(degrees)))]
        feats = pool[d]
        k = min(int(rng.integers(k_min, k_max + 1)), len(feats))
        w = np.array([1.0 / f.size for f in feats])
        idx = set(rng.choice(len(feats), size=k, replace=False, p=w / w.sum()).tolist())
        if d == 0:
            one = [i for i, f in enumerate(feats) if f.text == "1"]
            if one:
                idx.add(one[0])
        chosen = tuple(sorted((feats[i] for i in idx), key=lambda f: (f.size, f.text)))
        tasks.append(Task("positive3", chosen, seed=int(rng.integers(2 ** 31))))
    return tasks


def small_subset_tasks(pool: Dict[int, List[Feature]], sizes: Sequence[int], rng: np.random.Generator) -> List[Task]:
    """Exhaustive phase: every subset of the given sizes within each degree's pool.

    Random subsets of size 4-12 rarely contain a minimal known result's features, and an extreme ray
    needs k-1 independent tight rows, so small sets are enumerated outright.
    """
    tasks: List[Task] = []
    for d in sorted(pool):
        for k in sizes:
            for combo in itertools.combinations(pool[d], k):
                tasks.append(Task("positive3", tuple(combo), seed=int(rng.integers(2 ** 31))))
    return tasks


def run_task(task: Task, known: KnownIndex, grid_n: int) -> dict:
    """Stages 3-7 for one task. Returns plain data so it can cross a process boundary."""
    feats = list(task.features)
    exprs = [expr_of(f) for f in feats]
    kept, M = al.evaluate_table(exprs, al.grid_points(grid_n))
    out = {"degree": feats[0].degree if feats else None, "seed": task.seed, "features": [f.text for f in feats],
           "n_rows": len(kept), "status": "ok", "rels": []}
    if len(kept) < RULE_MIN_ROWS * len(feats):
        out["status"] = "too_few_rows"
        return out
    eqs, basis = al.find_equations(feats, exprs, M)
    rng = np.random.default_rng(task.seed)
    rels = al.inequalities(feats, exprs, basis, M, kept, rng)
    rels += al.target_bounds(feats, exprs, basis, M, kept)
    for e in eqs:
        out["rels"].append({"kind": "eq", "coeffs": list(e.coeffs), "features": [f.text for f in e.features],
                            "tight": [], "min_slack": "", "status": "proved", "score": 0.0,
                            "canon": "0", "known": "", "routine": True, "removed_by": "routine_equation"})
    seen_coeffs = set()
    for r in rels:
        if r.coeffs in seen_coeffs:
            continue
        seen_coeffs.add(r.coeffs)
        rel_feats = list(r.features)
        rel_exprs = [expr_of(f) for f in rel_feats]
        comb = combination(r.coeffs, rel_feats)
        removed = None
        if al.is_sign_trivial(r.coeffs, rel_exprs) or expanded_sign_trivial(comb):
            removed = "sign_trivial"
        elif is_loose(r):
            removed = "loose"
        status = r.status
        if removed is None:
            verdict, _ = level1(r.coeffs, rel_feats, np.random.default_rng(task.seed + len(out["rels"])))
            if verdict == "refuted":
                status, removed = "refuted", "refuted_level1"
        out["rels"].append({"kind": r.kind, "coeffs": list(r.coeffs), "features": [f.text for f in rel_feats],
                            "tight": list(r.tight_rows), "min_slack": r.min_slack, "status": status,
                            "score": score(r), "canon": canonical_key(comb),
                            "known": known.match(comb) or "", "routine": False, "removed_by": removed})
    return out


def _worker(args):
    """Process entry point. Boundary grid points give 0/0 in float lambdify; those rows are dropped upstream."""
    task, known_entries, grid_n = args
    with np.errstate(all="ignore"):
        return run_task(task, KnownIndex(known_entries), grid_n)
