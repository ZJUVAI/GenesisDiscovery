# AGENTS.md

Relation Discovery Engine (RDE): finds equations and inequalities among algebraic expressions
and integer relations among constants via continued fractions. See `RESULTS.md` and
`reports/RDE_article.md` for what has been found.

## Layout
- `rde/`: engine. `expr.py` (parser, canonical form), `algebra.py` (grid, equations, inequalities,
  proofs), `constants.py` (continued fractions, PSLQ), `types.py` (dataclasses).
- `tests/`: pytest suite. Must pass before any change is reported as done.
- `scripts/`: experiment drivers. Output goes to `results/`.
- `reports/`: write-ups. `RESULTS.md`: results summary.

## Commands
- Test: `PYTHONPATH=. python3 -m pytest -q tests`
- Run experiments: see the Reproduce section of `RESULTS.md`.

## Rules
- Never call `eval`/`exec` on user or generated text. Parse with `expr.parse` only.
- Accept/reject decisions use exact arithmetic (Fraction, sympy) or mpmath at stated precision.
  Float64 is only for finding candidates; always recheck.
- Do not loosen a tolerance, row count, or test fixture to make a test pass. Report the failure.
- Do not report a relation as new without a literature check. Say "unverified" otherwise.
- Do not commit or push unless asked.

## Environment
- Python 3.8 here (the original plan asked for 3.11): avoid `X | Y` type syntax at runtime, use
  `from __future__ import annotations`.
- Shared machine: use at most 48 worker processes, and set `OMP_NUM_THREADS=1`.
- Stop background jobs by PID (`ps`, then `kill`), not `pkill -f`, which can match your own shell.
