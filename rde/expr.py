"""Expression language for the algebraic domain: strict parser, canonical form, enumerator.

Text is parsed with the `ast` module and only a whitelist of node types is accepted.
Nothing is ever passed to eval() or exec().
"""
from __future__ import annotations

import ast
import itertools
from typing import Dict, Iterable, List, Optional, Tuple

import sympy as sp

from .types import Feature

A, B, C = sp.symbols("a b c")
VARS: Tuple[sp.Symbol, ...] = (A, B, C)
_NAMES = {"a": A, "b": B, "c": C}
_WRAPPERS = {"sym": list(itertools.permutations(VARS)), "cyc": [(A, B, C), (B, C, A), (C, A, B)]}
_FINGERPRINT_POINTS = [
    (sp.Rational(2, 7), sp.Rational(3, 11), sp.Rational(5, 13)),
    (sp.Rational(1, 3), sp.Rational(1, 5), sp.Rational(7, 9)),
    (sp.Rational(4, 3), sp.Rational(2, 9), sp.Rational(6, 5)),
    (sp.Rational(9, 4), sp.Rational(5, 2), sp.Rational(1, 6)),
    (sp.Rational(3, 8), sp.Rational(11, 7), sp.Rational(2, 3)),
    (sp.Rational(7, 5), sp.Rational(1, 2), sp.Rational(4, 11)),
    (sp.Rational(5, 6), sp.Rational(13, 10), sp.Rational(3, 17)),
    (sp.Rational(8, 3), sp.Rational(7, 4), sp.Rational(9, 19)),
]


def _int_const(node: ast.AST, lo: int, hi: int) -> int:
    """Return the value of an integer literal node within [lo, hi], else raise."""
    if isinstance(node, ast.Constant) and type(node.value) is int and lo <= node.value <= hi:
        return node.value
    raise ValueError(f"expected integer in [{lo}, {hi}]")


def _convert(node: ast.AST) -> sp.Expr:
    """Convert a whitelisted AST node to a sympy expression."""
    if isinstance(node, ast.Name):
        if node.id not in _NAMES:
            raise ValueError(f"unknown name {node.id!r}")
        return _NAMES[node.id]
    if isinstance(node, ast.Constant):
        return sp.Integer(_int_const(node, 1, 9))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_convert(node.operand)
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Pow):
            base = _convert(node.left)
            return base ** _int_const(node.right, 2, 4)
        left, right = _convert(node.left), _convert(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sqrt":
        if len(node.args) != 1 or node.keywords:
            raise ValueError("sqrt takes one argument")
        return sp.sqrt(_convert(node.args[0]))
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in _WRAPPERS:
        inner = node.slice.value if isinstance(node.slice, ast.Index) else node.slice
        return _wrap(node.value.id, _convert(inner))
    raise ValueError(f"unsupported syntax: {type(node).__name__}")


def _wrap(name: str, e: sp.Expr) -> sp.Expr:
    """Apply sym[...] or cyc[...]: sum over the distinct permuted copies of e."""
    seen: Dict[str, sp.Expr] = {}
    for perm in _WRAPPERS[name]:
        term = sp.cancel(e.subs(dict(zip(VARS, perm)), simultaneous=True))
        seen.setdefault(sp.srepr(term), term)
    return sp.Add(*seen.values())


def parse(text: str) -> sp.Expr:
    """Parse a feature string such as 'sym[a^2*b] + a*b*c' into a sympy expression."""
    source = text.strip().replace("^", "**")
    if len(source) > 400:
        raise ValueError("expression too long")
    try:
        tree = ast.parse(source, mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"syntax error in {text!r}") from exc
    return _convert(tree)


def _text_size(text: str) -> int:
    """Number of nodes in the syntax tree of the input text (the stored expression's size)."""
    return sum(1 for _ in ast.walk(ast.parse(text.strip().replace("^", "**"), mode="eval")))


def degree(f: sp.Expr) -> Optional[int]:
    """Homogeneous degree of a rational function, or None if not homogeneous."""
    num, den = sp.fraction(sp.cancel(f))
    homogeneous_degrees = []
    for part in (num, den):
        if part == 0:
            return None
        totals = {sum(m) for m in sp.Poly(part, *VARS).monoms()}
        if len(totals) != 1:
            return None
        homogeneous_degrees.append(totals.pop())
    return homogeneous_degrees[0] - homogeneous_degrees[1]


def _fingerprint(f: sp.Expr) -> Optional[Tuple[str, ...]]:
    """Scale-free numeric fingerprint: values at fixed points, divided by the first nonzero value."""
    vals = []
    for pt in _FINGERPRINT_POINTS:
        try:
            v = sp.N(f.subs(dict(zip(VARS, pt))), 40)
        except (ZeroDivisionError, TypeError):
            return None
        if not v.is_finite or v.is_real is False:
            return None
        vals.append(v)
    base = next((v for v in vals if v != 0), None)
    if base is None:
        return None
    return tuple(str(sp.N(v / base, 25)) for v in vals)


def canonical(text: str) -> Tuple[Feature, Tuple[str, ...]]:
    """Return the Feature for a text plus its fingerprint, used for deduplication."""
    expr = parse(text)
    f = sp.cancel(expr)
    fp = _fingerprint(f)
    if fp is None:
        raise ValueError(f"{text!r} is not finite at the fingerprint points")
    return Feature(text=text.strip(), size=_text_size(text), degree=degree(f)), fp


def _monomial_text(i: int, j: int, k: int) -> str:
    """Text of a|^i b^j c^k with exponent 1 written without a power."""
    parts = []
    for name, e in zip("abc", (i, j, k)):
        if e == 1:
            parts.append(name)
        elif e > 1:
            parts.append(f"{name}^{e}")
    return "*".join(parts) if parts else "1"


def enumerate_features(domain: str, max_degree: int, max_size: int = 40) -> Dict[int, List[Feature]]:
    """Enumerate deduplicated symmetric polynomial features, grouped by homogeneous degree.

    Candidates are sym[monomial] for every exponent triple, and products of two such
    sums (degree-additive). Duplicates are removed by fingerprint; the shortest text wins.
    """
    if domain != "positive3":
        raise ValueError("enumerate_features currently supports the positive3 domain only")
    atoms = []
    for d in range(0, max_degree + 1):
        for i in range(d, -1, -1):
            for j in range(d - i, -1, -1):
                k = d - i - j
                if k <= j:
                    atoms.append(f"sym[{_monomial_text(i, j, k)}]")
    candidates = list(atoms)
    for x, y in itertools.combinations_with_replacement(atoms, 2):
        candidates.append(f"{x}*{y}")
    best: Dict[Tuple[str, ...], Feature] = {}
    for text in candidates:
        try:
            feat, fp = canonical(text)
        except (ValueError, ZeroDivisionError):
            continue
        if feat.degree is None or feat.size > max_size or feat.degree > max_degree:
            continue
        old = best.get(fp)
        if old is None or (feat.size, feat.text) < (old.size, old.text):
            best[fp] = feat
    grouped: Dict[int, List[Feature]] = {}
    for feat in best.values():
        grouped.setdefault(feat.degree, []).append(feat)
    for feats in grouped.values():
        feats.sort(key=lambda f: (f.size, f.text))
    return grouped


def expr_of(feature: Feature) -> sp.Expr:
    """Sympy rational function for a Feature."""
    return sp.cancel(parse(feature.text))


def iter_terms(features: Iterable[Feature]) -> List[sp.Expr]:
    """Sympy expressions for a list of features, in order."""
    return [expr_of(f) for f in features]
