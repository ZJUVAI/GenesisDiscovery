"""M2-M4 acceptance for the algebraic domain: table, equations, inequalities, proofs."""
import numpy as np
import pytest
import sympy as sp

from rde import algebra as al
from rde.expr import enumerate_features, expr_of, parse


def _setup(d, texts):
    feats = enumerate_features("positive3", d)[d]
    chosen = [f for f in feats if f.text in texts]
    assert len(chosen) == len(texts), [f.text for f in feats]
    exprs = [expr_of(f) for f in chosen]
    pts = al.grid_points()
    kept, M = al.evaluate_table(exprs, pts)
    return chosen, exprs, kept, M


def test_integer_vector_rationalisation_example():
    assert al.to_integer_vector([0.5, -1, 0.3333333333], first_positive=True) == (3, -6, 2)


def test_grid_points_are_on_simplex_without_corners():
    pts = al.grid_points()
    assert all(sum(p) == 1 and min(p) >= 0 for p in pts)
    assert all(sorted(p)[:2] != [0, 0] for p in pts)


def test_pole_rows_are_dropped_without_exception():
    kept, M = al.evaluate_table([parse("1/(a-b)")], al.grid_points())
    assert len(kept) < len(al.grid_points())
    assert np.all(np.isfinite(M))


def test_identity_equations_are_recovered_exactly():
    feats = enumerate_features("positive3", 3)[3]
    exprs = [expr_of(f) for f in feats]
    kept, M = al.evaluate_table(exprs, al.grid_points())
    eqs, basis = al.find_equations(feats, exprs, M)
    assert eqs, "expected at least one identity among degree-3 features"
    for eq in eqs:
        total = sp.cancel(sum(c * expr_of(f) for c, f in zip(eq.coeffs, eq.features)))
        assert total == 0


def test_schur_degree_three_is_proved():
    chosen, exprs, kept, M = _setup(3, ["sym[a^3]", "sym[a*b^2]", "sym[a*b*c]"])
    rels = al.inequalities(chosen, exprs, [0, 1, 2], M, kept, np.random.default_rng(0))
    schur = [r for r in rels if r.coeffs in {(1, -1, 3), (-1, 1, -3)}]
    assert schur and schur[0].status == "proved"
    tight = set(schur[0].tight_rows)
    assert "(1/3, 1/3, 1/3)" in tight and "(1/2, 1/2, 0)" in tight


def test_amgm_style_sum_is_not_reported_separately():
    # sum a^3 - 3abc = Schur + (sym a^2 b - 6abc); it must not appear as an extreme ray
    chosen, exprs, kept, M = _setup(3, ["sym[a^3]", "sym[a*b^2]", "sym[a*b*c]"])
    rels = al.inequalities(chosen, exprs, [0, 1, 2], M, kept, np.random.default_rng(0))
    assert all(r.coeffs != (1, 0, -3) for r in rels)


def test_proof_rejects_a_false_inequality():
    a, b, c = al.VARS
    assert al.ordered_difference_proof(sp.expand(a**2 - 2 * a * b)) is None


def test_nesbitt_is_proved_with_rational_features():
    from rde.expr import canonical
    feats = [canonical(s)[0] for s in ["cyc[a/(b+c)]", "1"]]
    exprs = [expr_of(f) for f in feats]
    kept, M = al.evaluate_table(exprs, al.grid_points())
    rels = al.inequalities(feats, exprs, [0, 1], M, kept, np.random.default_rng(0))
    assert any(r.coeffs == (2, -3) and r.status == "proved" for r in rels)


def test_target_bound_finds_sym_a2b_at_least_6abc():
    # centre-tangent case: not an extreme ray, found by the target-bound mode (plan 8, second mode)
    chosen, exprs, kept, M = _setup(3, ["sym[a*b^2]", "sym[a*b*c]"])
    rels = al.target_bounds(chosen, exprs, [0, 1], M, kept)
    assert any(r.coeffs == (1, -6) and r.status == "proved" for r in rels)


def test_target_bound_keeps_weaker_sum_out_of_three_feature_run():
    # with sym a^3, sym a^2b, abc: Schur is found, and sum a^3 - 3abc (= Schur + (sym a^2b - 6abc)) is not
    chosen, exprs, kept, M = _setup(3, ["sym[a^3]", "sym[a*b^2]", "sym[a*b*c]"])
    rels = al.target_bounds(chosen, exprs, [0, 1, 2], M, kept)
    assert all(r.coeffs != (1, 0, -3) for r in rels)
    assert all(r.status == "proved" for r in rels)


def test_m3_square_of_sum_identity():
    # M3: (a+b+c)^2 = sym[a^2] + 2*sym[a*b] (sym[a*b] = ab+bc+ca)
    from rde.expr import canonical
    feats = [canonical(s)[0] for s in ["(a+b+c)^2", "sym[a^2]", "sym[a*b]"]]
    exprs = [expr_of(f) for f in feats]
    kept, M = al.evaluate_table(exprs, al.grid_points())
    eqs, _ = al.find_equations(feats, exprs, M)
    assert any(e.coeffs in {(1, -1, -2), (-1, 1, 2)} for e in eqs)
