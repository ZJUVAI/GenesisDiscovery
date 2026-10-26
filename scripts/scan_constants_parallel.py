"""Parallel scan of polynomial continued fractions (deg p, deg q <= 3) against named constants.

Each worker handles a chunk of (p, q) pairs. Results are verified at higher precision before saving.
"""
import json
import multiprocessing as mp_proc
import sys
import time

import mpmath as mp

from rde import constants as C

DEG_P, DEG_Q, COEFF = 3, 3, 3
DEPTH = 400
DPS = 100
MAXCOEFF = 1000
NAMES = ["pi", "e", "log2", "zeta3", "catalan", "zeta5"]


def _all_polys(deg, rng):
    """All coefficient tuples of length deg+1 with entries in rng."""
    if deg == 0:
        return [(c,) for c in rng]
    return [head + (c,) for head in _all_polys(deg - 1, rng) for c in rng]


def _work(chunk):
    """Evaluate a chunk of (p, q) pairs; return verified hits and the K values seen."""
    mp.mp.dps = DPS
    hits, seen = [], []
    for p, q in chunk:
        K = C.pcf_converged(p, q, DEPTH, DPS)
        if K is None or C.is_rational(K):
            continue
        key = mp.nstr(K, 50)
        seen.append(key)
        for name in NAMES:
            x = C.CONSTANTS[name]()
            rel = C.relation_search(K, x, dps=60, maxcoeff=MAXCOEFF)
            if rel is None:
                continue
            log_res = C.verify_relation(rel, p, q, name, dps=220)
            if log_res is None or log_res > -150:
                continue
            hits.append({"constant": name, "p": list(p), "q": list(q), "relation": list(rel),
                         "closed_form": C.describe_relation(rel, name), "log10_residual": round(log_res, 1),
                         "K": key[:30]})
    return hits, seen


def main():
    rng = range(-COEFF, COEFF + 1)
    pairs = []
    for p in _all_polys(DEG_P, rng):
        if any(C._poly_eval(p, n) == 0 for n in range(4)):
            continue
        for q in _all_polys(DEG_Q, rng):
            if any(q):
                pairs.append((p, q))
    print("pairs", len(pairs), flush=True)
    chunks = [pairs[i:i + 500] for i in range(0, len(pairs), 500)]
    hits, distinct = [], set()
    t0 = time.time()
    with mp_proc.get_context("fork").Pool(48) as pool:
        for n, (h, s) in enumerate(pool.imap_unordered(_work, chunks), 1):
            hits.extend(h)
            distinct.update(s)
            if n % 100 == 0:
                print(f"chunks={n}/{len(chunks)} hits={len(hits)} distinctK={len(distinct)} t={time.time()-t0:.0f}s", flush=True)
    with open("results/constants_deep.json", "w") as fh:
        json.dump({"deg_p": DEG_P, "deg_q": DEG_Q, "coeff": COEFF, "depth": DEPTH, "hits": hits,
                   "distinct_K": len(distinct)}, fh, indent=1)
    print("done hits", len(hits), "distinct K", len(distinct), "time", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
