"""Scan polynomial continued fractions against named constants and save verified integer relations."""
import json
import sys
import time

from rde import constants as C

max_coeff = int(sys.argv[1]) if len(sys.argv) > 1 else 5
out = sys.argv[2] if len(sys.argv) > 2 else "results/constants_hits.json"
names = list(C.CONSTANTS)


def progress(count, distinct, hits):
    """Print scan progress."""
    print(f"pairs={count} distinct_K={distinct} hits={hits} t={time.time() - t0:.0f}s", flush=True)


t0 = time.time()
print("M3 check:", C.check_known_4_over_pi(), flush=True)
hits = C.search(names, max_coeff, progress=progress)
hits.sort(key=lambda h: (h["constant"], sum(abs(c) for c in h["relation"])))
records = [{"constant": h["constant"], "p": list(h["p"]), "q": list(h["q"]),
            "relation": list(h["relation"]), "closed_form": h["closed_form"],
            "log10_residual": h["log10_residual"]} for h in hits]
with open(out, "w") as fh:
    json.dump({"max_coeff": max_coeff, "constants": names, "hits": records}, fh, indent=1)
print("done hits", len(records), "time", round(time.time() - t0), flush=True)
