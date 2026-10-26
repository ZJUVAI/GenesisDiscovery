"""Parallel scan of 4-feature subsets of a degree-0 rational pool for extreme inequalities.

Relations are deduplicated up to a positive scalar using the primitive numerator of the
combined rational function, so a relation and its rescalings count once.
"""
import itertools
import json
import multiprocessing as mp_proc
import time
import warnings

import numpy as np
import sympy as sp

warnings.filterwarnings("ignore")

from rde import algebra as al
from rde.expr import canonical, expr_of

POOL = [
    "1",
    "cyc[a/(b+c)]",
    "cyc[a^2/(b^2+c^2)]",
    "cyc[a^2/(b+c)^2]",
    "cyc[a/(a+b)]",
    "cyc[a/b]",
    "cyc[b/a]",
    "sym[a^3]/(a*b*c)",
    "sym[a^2]/sym[a*b]",
    "sym[a/b]",
    "cyc[a*b/c^2]",
    "cyc[(a^2+b*c)/(b^2+c^2)]",
    "cyc[a^3/(b^3+c^3)]",
    "cyc[a/(b^2+c^2)]",
    "cyc[(b+c)/a]",
    "cyc[a^2/(b*c)]",
    "cyc[a^2/(a^2+b*c)]",
    "sym[a*b]/sym[a^2]",
    "cyc[a^3/(a^2+b*c)]",
]
SUBSET = 4


def _key(coeffs, exprs):
    """Scale-free key: primitive numerator and denominator of sum c_i f_i, sign-normalised."""
    num, den = sp.fraction(sp.together(sum(c * e for c, e in zip(coeffs, exprs))))
    num = sp.Poly(sp.expand(num), *al.VARS)
    den = sp.Poly(sp.expand(den), *al.VARS)
    _, num = num.primitive()
    _, den = den.primitive()
    if num.LC() < 0:
        num = -num
    return str(num.as_expr()) + " / " + str(den.as_expr())


def _work(combo):
    """Run the cone analysis on one subset of the pool."""
    rng = np.random.default_rng(hash(combo) % (2 ** 32))
    try:
        fl = [canonical(t)[0] for t in combo]
        exprs = [expr_of(f) for f in fl]
        kept, M = al.evaluate_table(exprs, al.grid_points())
        if len(kept) < 50:
            return []
        eqs, basis = al.find_equations(fl, exprs, M)
        if not basis:
            return []
        out = []
        for r in al.inequalities(fl, exprs, basis, M, kept, rng, n_objectives=150):
            sub = [exprs[i] for i in basis]
            out.append({"subset": list(combo), "basis": [fl[i].text for i in basis], "coeffs": list(r.coeffs),
                        "status": r.status, "n_tight": len(r.tight_rows), "min_slack": r.min_slack,
                        "key": _key(r.coeffs, sub), "max_coeff": max(abs(c) for c in r.coeffs)})
        return out
    except Exception as exc:  # keep the scan going on a bad subset
        return [{"subset": list(combo), "error": repr(exc)}]


def main():
    combos = list(itertools.combinations(POOL, SUBSET))
    print("subsets", len(combos), flush=True)
    t0 = time.time()
    found = {}
    with mp_proc.get_context("fork").Pool(48) as pool:
        for n, res in enumerate(pool.imap_unordered(_work, combos, chunksize=2), 1):
            for r in res:
                if "error" in r:
                    continue
                prev = found.get(r["key"])
                if prev is None or (r["status"], -r["max_coeff"]) > (prev["status"], -prev["max_coeff"]):
                    found[r["key"]] = r
            if n % 500 == 0:
                print(f"subsets={n}/{len(combos)} distinct={len(found)} t={time.time()-t0:.0f}s", flush=True)
    rows = sorted(found.values(), key=lambda r: (r["status"] != "proved", r["max_coeff"]))
    with open("results/rational_deep.json", "w") as fh:
        json.dump(rows, fh, indent=1)
    print("done distinct", len(rows), "proved", sum(r["status"] == "proved" for r in rows),
          "time", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
