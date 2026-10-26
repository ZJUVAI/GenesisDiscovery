"""Run the algebraic pipeline on symmetric polynomial families and the rational degree-0 family."""
import json
import time

import numpy as np

from rde import algebra as al
from rde.expr import canonical, enumerate_features, expr_of

RATIONAL_FAMILY = ["cyc[a/(b+c)]", "1", "sym[a^2]/(sym[a*b])"]


def run_family(label, feats, rng):
    """Equations, basis, and inequality rays for one feature list."""
    exprs = [expr_of(f) for f in feats]
    kept, M = al.evaluate_table(exprs, al.grid_points())
    eqs, basis = al.find_equations(feats, exprs, M)
    t = time.time()
    rels = al.inequalities(feats, exprs, basis, M, kept, rng)
    return {
        "family": label,
        "features": [f.text for f in feats],
        "rows": len(kept),
        "equations": [{"coeffs": list(e.coeffs), "features": [f.text for f in e.features]} for e in eqs],
        "basis": [feats[i].text for i in basis],
        "inequalities": [{"coeffs": list(r.coeffs), "features": [f.text for f in r.features],
                          "status": r.status, "tight_points": list(r.tight_rows), "min_slack": r.min_slack}
                         for r in rels],
        "seconds": round(time.time() - t, 1),
    }


def main():
    rng = np.random.default_rng(20261008)
    results = []
    for d in (2, 3, 4):
        feats = enumerate_features("positive3", d)[d]
        results.append(run_family(f"symmetric polynomial, degree {d}", feats, rng))
        print(d, len(results[-1]["inequalities"]), "inequalities", flush=True)
    rational = [canonical(s)[0] for s in ["cyc[a/(b+c)]", "1", "sym[a^2]/sym[a*b]"]]
    results.append(run_family("rational, degree 0", rational, rng))
    with open("results/algebraic_results.json", "w") as fh:
        json.dump(results, fh, indent=1)
    for r in results:
        print(r["family"], "rows", r["rows"], "basis", r["basis"])
        for ineq in r["inequalities"]:
            print("   ", ineq["status"], ineq["coeffs"], ineq["features"], "tight", ineq["tight_points"])


if __name__ == "__main__":
    main()
