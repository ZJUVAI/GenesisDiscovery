"""M2 acceptance: domains and table cleaning (plan 5, 6, 12)."""
import numpy as np
import pytest

from rde.domains import Positive3
from rde.table import build_table
from rde.types import Feature, Task


def _f(text):
    return Feature(text=text, size=1, degree=None)


@pytest.fixture
def domain():
    return Positive3()


def test_all_sampled_rows_are_on_the_simplex(domain):
    rng = np.random.default_rng(1)
    rows = domain.sample(500, rng) + domain.special_points()
    assert all(domain.contains(r) for r in rows)


def test_pole_feature_is_removed_without_exception(domain):
    task = Task("positive3", (_f("1/(a-b)"), _f("sym[a^2]"), _f("sym[a*b]")), seed=0)
    table = build_table(task, domain, np.random.default_rng(0))
    assert "1/(a-b)" in table.dropped_features
    assert all(np.all(np.isfinite(row)) for row in table.M)


def test_equal_columns_are_reported(domain):
    task = Task("positive3", (_f("a*b+b*c+c*a"), _f("sym[a*b]"), _f("sym[a^2]")), seed=0)
    table = build_table(task, domain, np.random.default_rng(0))
    assert ("a*b+b*c+c*a", "sym[a*b]") in table.duplicates
    assert [f.text for f in table.features] == ["a*b+b*c+c*a", "sym[a^2]"]


def test_too_few_rows_is_a_named_failure(domain):
    from rde.table import TooFewRows
    task = Task("positive3", (_f("sym[a^2]"), _f("sym[a*b]")), seed=0)
    rows = [(1 / 3, 1 / 3, 1 / 3)] * 3
    with pytest.raises(TooFewRows):
        build_table(task, domain, np.random.default_rng(0), rows=rows)


def test_minimize_returns_points_on_the_simplex(domain):
    # regression: Nelder-Mead used to leave the unit square and return a point with a negative coordinate
    for seed in range(5):
        pt, _ = domain.minimize(lambda x: -x[0] * x[1] * x[2] + 1e-3 * (x[0] - 0.9) ** 2, 1,
                                np.random.default_rng(seed))
        assert domain.contains(pt)
