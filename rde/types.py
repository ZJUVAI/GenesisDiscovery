"""Shared data types for the relation discovery engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Feature:
    """A canonical closed-form expression in a domain's variables."""

    text: str
    size: int
    degree: Optional[int]


@dataclass(frozen=True)
class Relation:
    """An integer equation (kind 'eq') or inequality 'sum c_i f_i >= 0' (kind 'ineq')."""

    kind: str
    coeffs: Tuple[int, ...]
    features: Tuple[Feature, ...]
    tight_rows: Tuple = ()
    min_slack: str = ""
    status: str = "candidate"
    score: float = 0.0


@dataclass(frozen=True)
class Task:
    """One pipeline run: a domain key, a feature tuple of one degree, and a seed."""

    domain: str
    features: Tuple[Feature, ...]
    seed: int
