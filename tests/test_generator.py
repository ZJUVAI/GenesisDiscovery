"""M7 unit checks: text format, completion validity, novelty rule (plan 11). No model is loaded here."""
import os

import pytest

from rde.generator import data, sample

DB = "results/rde_algebraic3.sqlite"


def test_format_round_trip_is_sorted_by_size():
    texts = ["sym[a^3]", "a*b*c", "(a+b+c)^3"]
    comp = data.completion_for(data.sorted_texts(texts))
    assert comp.endswith("<end>")
    from rde.expr import canonical
    sizes = [canonical(t)[0].size for t in comp.splitlines()[0].split(" ; ")]
    assert sizes == sorted(sizes)


def test_valid_completion_is_parsed_and_deduplicated():
    text = "sym[a^3] ; sym[a^3] ; a*b*c ; sym[a^2*b] ; sym[a*b^2] ; (a+b+c)^3<end>"
    feats, key = sample.parse_completion(text, degree=3)
    assert len(feats) == 4  # sym[a*b^2] is the same feature as sym[a^2*b]
    assert len(key) == 4


@pytest.mark.parametrize("text, degree", [
    ("__import__('os') ; sym[a^3] ; a*b*c ; sym[a^2*b] ; sym[a]<end>", 3),  # not in the grammar
    ("sym[a^3] ; a*b*c ; sym[a^2*b]<end>", 3),                            # fewer than 4 distinct
    ("sym[a^3] ; a*b*c ; sym[a^2*b] ; sym[a^2]<end>", 3),                 # mixed degrees
    ("sym[a^3] ; a*b*c ; sym[a^2*b] ; sym[a^4]<end>", 3),                 # degree 4 in a degree-3 task
    ("<end>", 3),
])
def test_invalid_completions_are_rejected(text, degree):
    assert sample.parse_completion(text, degree) is None


@pytest.mark.skipif(not os.path.exists(DB), reason="needs the algebraic run database")
def test_novelty_rule_keeps_only_tasks_with_new_relations():
    exs = data.positives(DB)
    assert exs and all(e.reward != 0 for e in exs)
    assert all(e.completion.endswith("<end>") for e in exs)
    dup = data.with_top_fraction_duplicated(exs, 0.2)
    assert len(dup) == len(exs) + max(1, round(0.2 * len(exs)))
