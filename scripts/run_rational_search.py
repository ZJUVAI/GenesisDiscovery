"""Scan 3-feature subsets of a pool of degree-0 cyclic/symmetric rational features for extreme inequalities."""
import itertools
import json
import time

import numpy as np
import sympy as sp

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
    "cyc[a^3/(a*b*c)]",
    "sym[a^2]/sym[a*b]",
    "sym[a/b]",
    "cyc[a*b/(c^2)]",
    "cyc[(a^2+b*c)/(b^2+c^2)]",
]


def canonical_relation(coeffs, exprs):
    """A readable canonical form of sum c_i f_i (monic numerator over denominator), used for dedup."""
    P = sp.factor(sp.cancel(sum(c * e for c, e in zip(coeffs, exprs))))
    return str(P)


def main():
    rng = np.random.default_rng(7)
    feats = {}
    for t in POOL:
        try:
            f, _ = canonical(t)
            feats[t] = f
        except ValueError as exc:
            print("skip", t, exc)
    names = list(feats)
    seen = {}
    t0 = time.time()
    for combo in itertools.combinations(names, 3):
        fl = [feats[n] for n in combo]
        exprs = [expr_of(f) for f in fl]
        kept, M = al.evaluate_table(exprs, al.grid_points())
        if len(kept) < 50:
            continue
        eqs, basis = al.find_equations(fl, exprs, M)
        if not basis:
            continue
        for r in al.inequalities(fl, exprs, basis, M, kept, rng, n_objectives=150):
            sub_exprs = [exprs[i] for i in basis]
            key = canonical_relation(r.coeffs, sub_exprs)
            if key in seen:
                continue
            seen[key] = {"combo": list(combo), "basis": [fl[i].text for i in basis], "coeffs": list(r.coeffs),
                         "status": r.status, "tight_points": list(r.tight_rows)[:12],
                         "n_tight": len(r.tight_rows), "expression": key}
            print(r.status, r.coeffs, [fl[i].text for i in basis], "tight", len(r.tight_rows), flush=True)
    print("done", len(seen), "distinct relations in", round(time.time() - t0), "s", flush=True)
    with open("results/rational_search.json", "w") as fh:
        json.dump(list(seen.values()), fh, indent=1)


if __name__ == "__main__":
    main()
