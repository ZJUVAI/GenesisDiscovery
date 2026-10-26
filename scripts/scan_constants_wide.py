"""Widened polynomial-CF scan toward Ramanujan-Machine degrees (plan 5; /goal RM comparison).

Template: deg p <= 3 with coefficients in [-3, 3]; deg q <= 6 with coefficients in {-1, 0, 1}.
Reuses the worker of scan_constants_parallel. Uses at most 48 processes (AGENTS.md) and stops at a
wall-clock deadline, writing whatever was finished.

Apery's zeta(3) fraction (a(n) coefficients up to 51) is outside this box; it is kept as a known test.
"""
import json
import multiprocessing as mp_proc
import sys
import time

import mpmath as mp

from rde import constants as C
from rde.constants_mitm import Index, verify
from scripts import scan_constants_parallel as base

MITM_BOUND = 12  # coefficient box of the meet-in-the-middle screen (recovers 70/76 known hits; PSLQ 76/76)
_INDEX = None

DEG_P, DEG_Q = 3, 6
COEFF_P, COEFF_Q = 3, 1
WORKERS = 48
DEADLINE_S = 2 * 3600 - 300  # leave 5 minutes for writing results
OUT = "results/constants_wide_deg6_mitm.json"


def _index():
    """One sorted index per constant, built once per worker process."""
    global _INDEX
    if _INDEX is None:
        mp.mp.dps = base.DPS
        _INDEX = {n: Index(float(C.CONSTANTS[n]()), bound=MITM_BOUND) for n in base.NAMES}
    return _INDEX


def _work_mitm(chunk):
    """Same output as scan_constants_parallel._work, with the PSLQ screen replaced by the MITM lookup."""
    mp.mp.dps = base.DPS
    idx = _index()
    hits, seen = [], []
    for p, q in chunk:
        K = C.pcf_converged(p, q, base.DEPTH, base.DPS)
        if K is None or C.is_rational(K):
            continue
        key = mp.nstr(K, 50)
        seen.append(key)
        for name in base.NAMES:
            x = C.CONSTANTS[name]()
            for rel in idx[name].lookup(float(K)):
                if not verify(rel, x, K):
                    continue
                log_res = C.verify_relation(rel, p, q, name, dps=220)
                if log_res is None or log_res > -150:
                    continue
                hits.append({"constant": name, "p": list(p), "q": list(q), "relation": list(rel),
                             "closed_form": C.describe_relation(rel, name), "log10_residual": round(log_res, 1),
                             "K": key[:30]})
                break
    return hits, seen


def _polys(deg, bound):
    return base._all_polys(deg, range(-bound, bound + 1))


def main():
    pairs = []
    for p in _polys(DEG_P, COEFF_P):
        if any(base.C._poly_eval(p, n) == 0 for n in range(4)):
            continue
        for q in _polys(DEG_Q, COEFF_Q):
            if any(q):
                pairs.append((p, q))
    print("pairs", len(pairs), flush=True)
    chunks = [pairs[i:i + 500] for i in range(0, len(pairs), 500)]
    hits, distinct, done, t0 = [], set(), 0, time.time()
    pool = mp_proc.get_context("fork").Pool(WORKERS)
    try:
        for h, s in pool.imap_unordered(_work_mitm, chunks):
            hits.extend(h)
            distinct.update(s)
            done += 1
            if done % 200 == 0:
                print(f"chunks={done}/{len(chunks)} hits={len(hits)} t={time.time() - t0:.0f}s", flush=True)
            if time.time() - t0 > DEADLINE_S:
                print("deadline reached", flush=True)
                break
    finally:
        pool.terminate()
        pool.join()
    with open(OUT, "w") as fh:
        json.dump({"deg_p": DEG_P, "deg_q": DEG_Q, "coeff_p": COEFF_P, "coeff_q": COEFF_Q, "depth": base.DEPTH,
                   "pairs": len(pairs), "chunks_done": done, "chunks_total": len(chunks),
                   "hits": hits, "distinct_K": len(distinct)}, fh, indent=1)
    print("done chunks", done, "/", len(chunks), "hits", len(hits), "distinct K", len(distinct),
          "time", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
