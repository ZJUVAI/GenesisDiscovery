"""Plan 11: parsing and validity checks for model completions."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ..expr import canonical
from ..types import Feature

from .data import END

MIN_DISTINCT = 4  # plan 11: fewer than 4 distinct features after canonicalisation is invalid


def parse_completion(text: str, degree: int) -> Optional[Tuple[List[Feature], Tuple]]:
    """Features and the canonical task key, or None if the completion is not a valid task of this degree.

    Validity (plan 11): every feature parses through expr.parse, all degrees equal `degree`, and at least
    4 distinct features remain after canonicalisation. Duplicate fingerprints are merged.
    """
    body = text.split(END)[0].strip()
    if not body:
        return None
    line = body.splitlines()[0].strip()
    feats, fps = [], {}
    for part in (p.strip() for p in line.split(";")):
        if not part:
            return None
        try:
            feat, fp = canonical(part)
        except (ValueError, ZeroDivisionError, TypeError, RecursionError):
            return None
        if feat.degree != degree:
            return None
        if fp not in fps:
            fps[fp] = feat
    if len(fps) < MIN_DISTINCT:
        return None
    feats = sorted(fps.values(), key=lambda f: (f.size, f.text))
    return feats, tuple(sorted(fps))


def task_key(features: Sequence[Feature]) -> Tuple:
    """Canonical task key used for the seen-task check (sorted fingerprints)."""
    return tuple(sorted(canonical(f.text)[1] for f in features))
