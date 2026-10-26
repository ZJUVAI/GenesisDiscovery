"""M1 acceptance: expression parsing, canonical form, degree, enumeration."""
import pytest

from rde.expr import canonical, degree, enumerate_features, expr_of, parse


def test_commutative_products_are_one_feature():
    assert canonical("a*b")[1] == canonical("b*a")[1]


def test_scalar_multiples_are_one_feature():
    assert canonical("2*sym[a*b]")[1] == canonical("sym[a*b]")[1]


def test_parser_rejects_code_and_unknown_names():
    for bad in ["__import__('os')", "a + d", "open('x')", "a.b"]:
        with pytest.raises(ValueError):
            parse(bad)


def test_degree_of_homogeneous_and_inhomogeneous():
    assert degree(expr_of(canonical("sym[a^2*b]")[0])) == 3
    assert degree(expr_of(canonical("a/(b+c)")[0])) == 0
    assert degree(expr_of(canonical("a+b*c")[0])) is None


@pytest.mark.parametrize("d, minimum", [(2, 2), (3, 3), (4, 4)])
def test_symmetric_polynomial_count(d, minimum):
    feats = enumerate_features("positive3", d)[d]
    symmetric_monomial_sums = [f for f in feats if f.text.count("sym[") == 1 and f.text.startswith("sym[")]
    assert len(symmetric_monomial_sums) >= minimum
