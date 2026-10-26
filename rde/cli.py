"""Command line: `python -m rde.cli run --config configs/algebraic3.yaml [--tasks N]`."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from collections import Counter

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from . import pipeline as pl  # noqa: E402
from . import store  # noqa: E402
from .filters import KnownIndex, implied_by, texts_combination  # noqa: E402

MAX_WORKERS = 48


def _known_entries():
    """Benchmark inequalities (tests/known_results.py); identities are not matched."""
    from tests.known_results import ALGEBRA
    return [e for e in ALGEBRA if e[1] == "ineq"]


def execute(tasks, known_entries, grid_n: int, workers: int) -> list:
    """Run run_task on each task, in order; returns the per-task result dicts."""
    workers = min(int(workers), MAX_WORKERS)
    args = [(t, known_entries, grid_n) for t in tasks]
    if workers > 1:
        with mp.get_context("spawn").Pool(workers) as p:
            return list(p.imap(pl._worker, args, chunksize=1))
    return [pl._worker(a) for a in args]


def persist(results: list, db: str, domain: str, seen: set = None, stored: list = None) -> Counter:
    """Dedupe (filter 1), then remove implied relations (filter 3), then write.

    `seen` (canonical keys) and `stored` (kept inequality expressions) carry across rounds.
    """
    seen = set() if seen is None else seen
    stored = [] if stored is None else stored
    conn = store.open_db(db)
    counts = Counter()
    for task_id, res in enumerate(results, start=1):
        rows = []
        for r in res["rels"]:
            is_new = r["canon"] not in seen
            if r["kind"] == "ineq" and r["removed_by"] is None:
                if not is_new:
                    r["removed_by"] = "duplicate"
                else:
                    expr = texts_combination(r["coeffs"], r["features"])
                    if implied_by(expr, stored):
                        r["removed_by"] = "implied_in_db"
                    else:
                        stored.append(expr)
                seen.add(r["canon"])
            r["is_new"] = is_new
            counts[r["removed_by"] or ("routine" if r["routine"] else "kept")] += 1
            rows.append(r)
        task_row = {"domain": domain, "degree": res["degree"], "seed": res["seed"],
                    "features": res["features"], "n_rows": res["n_rows"], "status": res["status"]}
        store.write_task(conn, task_id, task_row, rows)
    conn.close()
    return counts


def run(config: dict, n_tasks: int = None) -> dict:
    """Propose, run, dedupe in task order, write the database and a summary."""
    n = n_tasks or config["n_tasks"]
    rng = np.random.default_rng(config["seed"])
    pool = pl.build_pool(config["max_degree"], config["extra_features"])
    tasks = pl.small_subset_tasks(pool, config.get("exhaustive_sizes", []), rng)
    tasks += pl.baseline_tasks(pool, n, rng, config["k_min"], config["k_max"])
    results = execute(tasks, _known_entries(), config["grid_n"], config["workers"])
    counts = persist(results, config["db"], config["domain"])
    kept = Counter(r["status"] for res in results for r in res["rels"]
                   if r["kind"] == "ineq" and r["removed_by"] is None)
    summary = {"tasks": len(tasks), "random_tasks": n, "exhaustive_tasks": len(tasks) - n,
               "pool_sizes": {str(d): len(v) for d, v in pool.items()},
               "task_status": dict(Counter(res["status"] for res in results)),
               "relation_counts": dict(counts), "kept_status": dict(kept),
               "known_matched": sorted({r["known"] for res in results for r in res["rels"] if r["known"]})}
    if config.get("summary"):
        with open(config["summary"], "w") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
    return summary


def main(argv=None) -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(prog="rde")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="run one configured task batch")
    run_p.add_argument("--config", required=True)
    run_p.add_argument("--tasks", type=int, default=None)
    args = ap.parse_args(argv)
    with open(args.config) as fh:
        config = yaml.safe_load(fh)
    summary = run(config, args.tasks)
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
