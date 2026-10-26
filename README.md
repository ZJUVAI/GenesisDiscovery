# GenesisDiscovery

Experiments in automated mathematical discovery. The main project is the **Relation Discovery
Engine (RDE)**, which searches for equations and inequalities among algebraic expressions, and
for integer relations among constants via continued fractions (in the spirit of the Ramanujan
Machine).

## Status

Early, proof-of-concept stage. The engine rediscovers known results (Schur's inequality, Nesbitt's
inequality, classical continued fractions for e, π and log 2) and produces a few candidate
inequalities that are not yet checked against the literature.

## Quick start

```
PYTHONPATH=. python3 -m pytest -q tests
PYTHONPATH=. python3 scripts/run_algebraic.py
PYTHONPATH=. python3 scripts/run_constants.py 5 results/constants_hits.json
```

Requires Python 3 with mpmath, sympy, numpy, scipy and pytest. The large scans
(`scripts/scan_*_parallel.py`) take hours and are meant for a multi-core machine.

## Layout

- `rde/`: the engine (expression language, algebra, constants search).
- `scripts/`: experiment drivers. Results are written to `results/`.
- `tests/`: automated tests.

See `AGENTS.md` for conventions when working on the code.
