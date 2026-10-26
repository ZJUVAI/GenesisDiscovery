"""Plan 11: text format and training set from the result database (round 0 and later rounds)."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import List

from ..expr import canonical

END = "<end>"


@dataclass(frozen=True)
class Example:
    """One positive task: prompt, completion (features sorted by size), and its reward."""

    prompt: str
    completion: str
    reward: float


def prompt_for(domain: str, degree: int, k: int) -> str:
    """First line of the format in plan 11."""
    return f"<task> domain={domain} degree={degree} k={k}\n"


def completion_for(feature_texts: List[str]) -> str:
    """Second line: features joined by ' ; ', then the end marker."""
    return " ; ".join(feature_texts) + "\n" + END


def sorted_texts(texts: List[str]) -> List[str]:
    """Features ordered by (size, text) so the target is deterministic (plan 11)."""
    return [f.text for f in sorted((canonical(t)[0] for t in texts), key=lambda f: (f.size, f.text))]


def positives(db_path: str) -> List[Example]:
    """Tasks with at least one kept relation that is new to the database when the task ran (novelty rule)."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT t.degree, t.domain, t.features, MAX(r.score) FROM tasks t JOIN relations r ON r.task_id = t.task_id "
        "WHERE r.kind = 'ineq' AND r.removed_by IS NULL AND r.is_new = 1 AND r.status IN ('numeric', 'proved') "
        "GROUP BY t.task_id ORDER BY t.task_id").fetchall()
    conn.close()
    out = []
    for degree, domain, feats, reward in rows:
        texts = sorted_texts(json.loads(feats))
        out.append(Example(prompt_for(domain, degree, len(texts)), completion_for(texts), float(reward)))
    return out


def with_top_fraction_duplicated(examples: List[Example], fraction: float = 0.2) -> List[Example]:
    """Plan 11 sample weight: the top 20% of tasks by reward appear twice."""
    if not examples:
        return []
    ranked = sorted(examples, key=lambda e: -e.reward)
    top = {id(e) for e in ranked[: max(1, round(fraction * len(ranked)))]}
    return examples + [e for e in examples if id(e) in top]
