"""Execution accuracy: compare the rows a generated query returned with the gold rows.

The SQL text is never compared (many queries are correct); the results are:
- Column names are ignored, and the prediction may have extra columns: every gold
  column must match a different predicted column, value by value. ("Mobile share"
  answered with both the count and the percentage is still correct.)
- Row order matters only when the case says so (the question asks for a ranking
  or "latest"); otherwise rows are compared as a multiset.
- The number of rows must be equal, except for "which X is the most ..." cases
  (`top_k`): then the first k predicted rows must match the k gold rows, so a
  query that returns the whole ranking with the right item on top also passes.
- Numbers match when they are equal within 1e-6, or equal after rounding to two
  decimals (the precision the prompt asks for). Decimal and int compare as numbers.
- A datetime at midnight equals the same date; NULL equals NULL; text is exact.
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import date, datetime
from decimal import Decimal
from itertools import permutations
from typing import Any

Row = tuple[Any, ...]


def _normalize(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float, Decimal)):
        return float(value)
    if isinstance(value, datetime):
        return value.date() if value.time() == datetime.min.time() else value
    return value


def values_match(predicted: Any, gold: Any) -> bool:
    p, g = _normalize(predicted), _normalize(gold)
    if isinstance(p, float) and isinstance(g, float):
        if math.isclose(p, g, rel_tol=1e-6, abs_tol=1e-6):
            return True
        return abs(p - g) <= 0.005 + 1e-9  # equal at two decimals
    if isinstance(p, date) and isinstance(g, date):
        return p == g
    return type(p) is type(g) and p == g


def _column(rows: list[Row], index: int) -> list[Any]:
    return [row[index] for row in rows]


def _rows_match(predicted: list[Row], gold: list[Row], ordered: bool) -> bool:
    if len(predicted) != len(gold):
        return False
    if ordered:
        return all(
            all(values_match(p, g) for p, g in zip(prow, grow, strict=True))
            for prow, grow in zip(predicted, gold, strict=True)
        )
    # Unordered: each gold row takes the first predicted row that matches it. With
    # tolerance-based equality this greedy pass could in theory miss a valid pairing
    # when two rows differ by less than the tolerance; the gold sets have no such rows.
    remaining = list(predicted)
    for grow in gold:
        for index, prow in enumerate(remaining):
            if all(values_match(p, g) for p, g in zip(prow, grow, strict=True)):
                del remaining[index]
                break
        else:
            return False
    return True


def results_match(
    predicted_rows: list[Row],
    gold_rows: list[Row],
    *,
    ordered: bool = False,
    top_k: int | None = None,
) -> bool:
    if top_k is not None:
        return len(gold_rows) == top_k and results_match(
            predicted_rows[:top_k], gold_rows, ordered=True
        )
    if not gold_rows:
        return not predicted_rows
    if not predicted_rows or len(predicted_rows) != len(gold_rows):
        return False
    gold_width, predicted_width = len(gold_rows[0]), len(predicted_rows[0])
    if predicted_width < gold_width:
        return False
    # Cheap pre-filter: for each gold column, which predicted columns hold the
    # same values (as a multiset of rounded values)?
    candidates = []
    for g in range(gold_width):
        gold_values = _column(gold_rows, g)
        options = [
            p
            for p in range(predicted_width)
            if _same_multiset(_column(predicted_rows, p), gold_values)
        ]
        if not options:
            return False
        candidates.append(options)
    for mapping in permutations(range(predicted_width), gold_width):
        if any(mapping[g] not in candidates[g] for g in range(gold_width)):
            continue
        projected = [tuple(row[p] for p in mapping) for row in predicted_rows]
        if _rows_match(projected, gold_rows, ordered):
            return True
    return False


def _same_multiset(predicted: list[Any], gold: list[Any]) -> bool:
    def key(value: Any) -> Any:
        value = _normalize(value)
        if isinstance(value, float):
            return ("n", round(value, 2))
        return (type(value).__name__, value)

    if Counter(map(key, predicted)) == Counter(map(key, gold)):
        return True
    # Rounding at the .005 boundary can put equal-enough numbers in different
    # buckets; let the exact row check decide in that case.
    return all(isinstance(_normalize(v), float) for v in predicted + gold) and len(
        predicted
    ) == len(gold)
