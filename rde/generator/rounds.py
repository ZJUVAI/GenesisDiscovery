"""Plan 11 iteration: round 0 from the baseline, then sample, run, retrain. Logs per round to JSON."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter

import numpy as np
import yaml

from .. import cli, pipeline as pl, store
from ..types import Task
from . import data, sample, train

ROUNDS = 3


def _log_round(name, n_sampled, n_parse, n_seen, kept, new, n_run):
    """One log line per round: parse rate, share already seen, kept and new relations per 1000 tasks."""
    return {"round": name, "sampled": n_sampled, "parsed": n_parse,
            "parse_rate": round(n_parse / max(n_sampled, 1), 4),
            "seen_share": round(n_seen / max(n_parse, 1), 4), "run": n_run,
            "kept_per_1000": round(1000 * kept / max(n_run, 1), 2),
            "new_per_1000": round(1000 * new / max(n_run, 1), 2)}


def _stats(results, db, seen_relations):
    """Persist results and return (kept, new) counts for the round."""
    counts = cli.persist(results, db, "positive3", seen_relations)
    kept = sum(1 for res in results for r in res["rels"] if r["kind"] == "ineq" and r["removed_by"] is None)
    new = sum(1 for res in results for r in res["rels"] if r["kind"] == "ineq" and r["removed_by"] is None and r["is_new"])
    return kept, new, counts


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rde-generator")
    ap.add_argument("--config", default="configs/generator.yaml")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    os.makedirs(cfg["out_dir"], exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    pyrng = random.Random(cfg["seed"])
    pool = pl.build_pool(cfg["max_degree"], cfg["extra_features"])
    degrees = [d for d in sorted(pool) if len(pool[d]) >= pl.MIN_POOL]
    known = cli._known_entries()
    seen_relations, seen_tasks = set(), set()
    examples, log = [], []

    # round 0: baseline proposals, no model
    base_tasks = pl.baseline_tasks(pool, cfg["n_per_round"], rng, cfg["k_min"], cfg["k_max"])
    t0 = time.time()
    res0 = cli.execute(base_tasks, known, cfg["grid_n"], cfg["workers"])
    db0 = os.path.join(cfg["out_dir"], "round0.sqlite")
    for t in base_tasks:
        seen_tasks.add(sample.task_key(t.features))
    kept, new, _ = _stats(res0, db0, seen_relations)
    log.append(_log_round("round0_baseline", len(base_tasks), len(base_tasks), 0, kept, new, len(base_tasks)))
    log[-1]["seconds"] = round(time.time() - t0, 1)
    examples += data.positives(db0)
    dbs = [db0]

    for r in range(1, ROUNDS + 1):
        train_dir = os.path.join(cfg["out_dir"], f"adapter_round{r - 1}")
        tok, model, hist = train.train(train.BASE, examples, "cuda", train_dir, seed=cfg["seed"] + r,
                                       epochs=cfg["epochs"], log=lambda m: print(f"[round {r - 1}] {m}", flush=True))
        prompts, degs = [], []
        for _ in range(cfg["n_per_round"]):
            d = degrees[pyrng.randrange(len(degrees))]
            k = pyrng.randint(cfg["k_min"], cfg["k_max"])
            degs.append(d)
            prompts.append(data.prompt_for(cfg["domain"], d, k))
        t0 = time.time()
        texts = train.sample(tok, model, prompts, "cuda", seed=cfg["seed"] + 100 * r)
        del model
        import torch
        torch.cuda.empty_cache()
        tasks, n_parse, n_seen = [], 0, 0
        for text, d in zip(texts, degs):
            parsed = sample.parse_completion(text, d)
            if parsed is None:
                continue
            n_parse += 1
            feats, _ = parsed
            key = sample.task_key(feats)
            if key in seen_tasks:
                n_seen += 1
                continue
            seen_tasks.add(key)
            tasks.append(Task(cfg["domain"], tuple(feats), seed=pyrng.randrange(2 ** 31)))
        res = cli.execute(tasks, known, cfg["grid_n"], cfg["workers"])
        db = os.path.join(cfg["out_dir"], f"round{r}.sqlite")
        kept, new, _ = _stats(res, db, seen_relations)
        dbs.append(db)
        entry = _log_round(f"round{r}", len(texts), n_parse, n_seen, kept, new, len(tasks))
        entry["val_loss"] = hist[-1]["val_loss"]
        entry["seconds"] = round(time.time() - t0, 1)
        log.append(entry)
        examples = sum((data.positives(d) for d in dbs), [])
        print(json.dumps(entry), flush=True)
        with open(os.path.join(cfg["out_dir"], "generator_log.json"), "w") as fh:
            json.dump(log, fh, indent=2)
    return log


if __name__ == "__main__":
    main()
