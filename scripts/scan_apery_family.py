"""Large-coefficient search for zeta(3)-type fractions: p(n) of degree 3 with coefficients in [-B, B],
q(n) = s n^6 for s in a list (plan 5; Apery's fraction is p = (5, 27, 51, 34), s = -1).

Pipeline: vectorised float64 CF (rde.cf_vector) -> sorted zeta(3) Mobius index (rde.constants_mitm)
-> float residual filter -> 100-digit re-check (rde.constants.verify_relation). Uses at most 48 processes
and stops at a wall-clock deadline.
"""
import json
import multiprocessing as mp_proc
import os
import sys
import time

import mpmath as mp
import numpy as np

from rde import constants as C
from rde.cf_vector import cf_float
from rde.constants_mitm import Index, verify

DEPTH = 120
TOL = 1e-10          # float screen tolerance on the Mobius residual (relative to scale 1)
MITM_BOUND = 12
DPS = 100
WORKERS = 48
_IDX = None


def _index():
    """ζ(3) Mobius index, built once per worker."""
    global _IDX
    if _IDX is None:
        _IDX = Index(float(mp.zeta(3)), bound=MITM_BOUND)
    return _IDX


def accept(p, q, rel):
    """Log10 residual of a verified relation for zeta(3), or None. Rejects K ~ 0 (every (0, 0, c2, c3) holds)."""
    with mp.workdps(DPS):
        x = mp.zeta(3)
        Kmp = C.pcf_converged(p, q, 400, DPS)
        if Kmp is None or abs(Kmp) < mp.mpf(10) ** -30 or not verify(rel, x, Kmp, dps=60):
            return None
    log_res = C.verify_relation(rel, p, q, "zeta3", dps=220)
    if log_res is None or log_res >= -150:
        return None
    return log_res


def _task(args):
    """All (a2, a1, a0) for one a3 and one family s; returns verified hits and screen counts."""
    B, s, a3 = args
    mp.mp.dps = DPS  # x must be computed at working precision, not the 15-digit default
    idx = _index()
    r = np.arange(-B, B + 1, dtype=float)
    a2, a1, a0 = np.meshgrid(r, r, r, indexing="ij")
    a0, a1, a2 = a0.ravel(), a1.ravel(), a2.ravel()
    A = np.stack([a0, a1, a2, np.full_like(a0, float(a3))], axis=1)
    Bq = np.zeros((A.shape[0], 7))
    Bq[:, 6] = s
    K = cf_float(A, Bq, DEPTH)
    finite = np.isfinite(K)
    lo = np.searchsorted(idx.vals, K - 1e-9)
    hi = np.searchsorted(idx.vals, K + 1e-9)
    cand = np.nonzero(finite & (hi > lo))[0]
    hits, n_checked = [], 0
    for i in cand:
        p = tuple(int(v) for v in A[i])
        q = tuple(int(v) for v in Bq[i])
        x = mp.zeta(3)
        for rel in idx.coeffs[lo[i]:hi[i]]:
            rel = tuple(int(c) for c in rel)
            if abs(rel[0] + rel[1] * float(x) + rel[2] * K[i] + rel[3] * float(x) * K[i]) > TOL:
                continue
            n_checked += 1
            log_res = accept(p, q, rel)
            if log_res is not None:
                hits.append({"constant": "zeta3", "p": list(p), "q": list(q), "relation": list(rel),
                             "log10_residual": round(log_res, 1)})
                break
    return {"B": B, "s": s, "a3": a3, "screened": int(cand.size), "mp_checked": n_checked,
            "hits": hits}


def main(B=60, families=(-1, 1), deadline_s=None, out="results/apery_family.json"):
    """Run all (family, a3) tasks; write the hits and per-task screen statistics."""
    tasks = [(B, s, a3) for s in families for a3 in range(-B, B + 1)]
    deadline_s = deadline_s or 3600
    t0 = time.time()
    results = []
    pool = mp_proc.get_context("fork").Pool(min(WORKERS, os.cpu_count() or WORKERS))
    try:
        for res in pool.imap_unordered(_task, tasks):
            results.append(res)
            if len(results) % 20 == 0:
                hits = sum(len(r["hits"]) for r in results)
                print(f"tasks={len(results)}/{len(tasks)} hits={hits} t={time.time() - t0:.0f}s", flush=True)
            if time.time() - t0 > deadline_s:
                print("deadline reached", flush=True)
                break
    finally:
        pool.terminate()
        pool.join()
    hits = [h for r in results for h in r["hits"]]
    apery = any(h["p"] == [5, 27, 51, 34] and h["q"] == [0, 0, 0, 0, 0, 0, -1] for h in hits)
    with open(out, "w") as fh:
        json.dump({"B": B, "families": list(families), "tasks_done": len(results), "tasks_total": len(tasks),
                   "screened": sum(r["screened"] for r in results),
                   "mp_checked": sum(r["mp_checked"] for r in results),
                   "apery_recovered": apery, "hits": hits}, fh, indent=1)
    print("done tasks", len(results), "/", len(tasks), "hits", len(hits), "apery", apery,
          "time", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    fams = tuple(int(v) for v in sys.argv[2].split(",")) if len(sys.argv) > 2 else (-1, 1)
    out = sys.argv[3] if len(sys.argv) > 3 else "results/apery_family.json"
    main(B, fams, deadline_s=float(os.environ.get("DEADLINE_S", 3600)), out=out)
