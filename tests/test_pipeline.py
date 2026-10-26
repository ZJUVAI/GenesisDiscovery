"""M5 acceptance: pipeline on known cases, filters, and determinism (plan 12)."""
import numpy as np
import sympy as sp

from rde import algebra as al
from rde import cli
from rde import pipeline as pl
from rde.expr import canonical, parse
from rde.filters import KnownIndex, canonical_key, combination
from rde.types import Task
from tests.known_results import ALGEBRA


def _task(texts, seed=0):
    feats = tuple(canonical(t)[0] for t in texts)
    return Task("positive3", feats, seed)


def _proved(res, coeffs):
    return any(r["coeffs"] == list(coeffs) and r["status"] == "proved" and r["removed_by"] is None
               for r in res["rels"])


def test_amgm_reaches_proved():
    known = KnownIndex([e for e in ALGEBRA if e[1] == "ineq"])
    res = pl.run_task(_task(["sym[a^3]", "a*b*c"]), known, grid_n=30)
    assert _proved(res, (1, -3))


def test_schur_reaches_proved_and_is_matched_to_benchmark():
    known = KnownIndex([e for e in ALGEBRA if e[1] == "ineq"])
    res = pl.run_task(_task(["sym[a^3]", "sym[a^2*b]", "a*b*c"]), known, grid_n=30)
    assert _proved(res, (1, -1, 3))
    schur = [r for r in res["rels"] if r["coeffs"] == [1, -1, 3]][0]
    assert schur["known"] == "Schur t=1"


def test_nesbitt_reaches_proved():
    known = KnownIndex([e for e in ALGEBRA if e[1] == "ineq"])
    res = pl.run_task(_task(["cyc[a/(b+c)]", "1"]), known, grid_n=30)
    assert _proved(res, (2, -3))
    nes = [r for r in res["rels"] if r["coeffs"] == [2, -3]][0]
    assert nes["known"] == "Nesbitt"


def test_sign_trivial_relation_is_flagged():
    exprs = [parse("sym[a*b]")]
    assert al.is_sign_trivial((1,), exprs)


def test_canonical_key_ignores_positive_scale_and_permutation():
    k1 = canonical_key(sp.cancel(parse("sym[a^2] - sym[a*b]")))
    k2 = canonical_key(sp.cancel(parse("2*(sym[a^2] - sym[a*b])")))
    k3 = canonical_key(sp.cancel(parse("sym[a^2] - sym[a*b]").subs({sp.Symbol("a"): sp.Symbol("b"),
                                                                    sp.Symbol("b"): sp.Symbol("a")},
                                                                   simultaneous=True)))
    assert k1 == k2 == k3
    assert canonical_key(sp.cancel(parse("sym[a^2] - sym[a*b]"))) != canonical_key(
        sp.cancel(parse("sym[a*b] - sym[a^2]")))


def test_run_is_deterministic_for_one_seed(tmp_path):
    config = {"domain": "positive3", "max_degree": 3, "extra_features": ["cyc[a/(b+c)]", "1"],
              "n_tasks": 3, "k_min": 4, "k_max": 6, "seed": 7, "workers": 1, "grid_n": 20,
              "db": str(tmp_path / "a.sqlite"), "summary": None}
    from rde import store
    cli.run(config)
    first = store.dump(store.sqlite3.connect(config["db"]))
    config["db"] = str(tmp_path / "b.sqlite")
    cli.run(config)
    second = store.dump(store.sqlite3.connect(config["db"]))
    assert first == second


def test_combination_matches_hand_expression():
    feats = [canonical("cyc[a/(b+c)]")[0], canonical("1")[0]]
    assert sp.cancel(combination((2, -3), feats) - parse("2*cyc[a/(b+c)] - 3")) == 0


def test_titu_matches_benchmark_up_to_scale():
    # 2*cyc[a^2/(b+c)] - sym[a] is 2 * (cyc[a^2/(b+c)] - (a+b+c)/2): same inequality, different scale
    known = KnownIndex([e for e in ALGEBRA if e[1] == "ineq"])
    expr = combination((-1, 2), [canonical("sym[a]")[0], canonical("cyc[a^2/(b+c)]")[0]])
    assert known.match(expr) == "Titu: sum a^2/(b+c) >= (sum a)/2"


def test_expanded_sign_trivial_catches_abc_in_disguise():
    from rde.filters import expanded_sign_trivial
    expr = sp.cancel(parse("(a+b+c)*sym[a*b] - sym[a*b^2]"))
    assert expanded_sign_trivial(expr)
    assert not expanded_sign_trivial(sp.cancel(parse("sym[a^3] - sym[a^2*b] + 3*a*b*c")))


def test_level1_refutes_a_false_inequality():
    from rde.verify import level1
    feats = [canonical("sym[a^2]")[0], canonical("sym[a*b]")[0]]
    # sum a^2 - 2 sum ab >= 0 is false (a = b = c gives -3a^2)
    status, _ = level1((1, -2), feats, np.random.default_rng(0))
    assert status == "refuted"


def test_level1_keeps_a_true_inequality():
    from rde.verify import level1
    feats = [canonical("sym[a^3]")[0], canonical("a*b*c")[0]]
    status, _ = level1((1, -3), feats, np.random.default_rng(0))
    assert status == "numeric"


def test_implied_by_stored_nonnegative_combination():
    from rde.filters import implied_by
    s1 = parse("sym[a^3] - 3*a*b*c")       # AM-GM
    s2 = parse("sym[a^2*b] - 6*a*b*c")     # sym a^2 b >= 6abc
    assert implied_by(sp.cancel(s1 + s2), [s1, s2])
    # Schur = s1 - s2 is not a nonnegative combination of s1 and s2
    schur = sp.cancel(parse("sym[a^3] - sym[a^2*b] + 3*a*b*c"))
    assert not implied_by(schur, [s1, s2])
