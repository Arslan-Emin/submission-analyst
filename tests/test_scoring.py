"""Execution-accuracy rule tests: what counts as the same result."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import pytest

from evals.scoring import results_match, values_match

GOLD = [("Mobile", 198), ("Desktop", 152), ("Tablet", 50)]


@pytest.mark.parametrize(
    ("predicted", "gold", "expected"),
    [
        (59.05, 59.047619, True),  # same at two decimals
        (Decimal("59.05"), 59.05, True),
        (198, 198.0, True),
        (59.1, 59.05, False),
        (0.5905, 59.05, False),  # fraction is not a percentage
        (datetime(2026, 10, 19), date(2026, 10, 19), True),
        (datetime(2026, 10, 19, 9, 30), date(2026, 10, 19), False),
        (None, None, True),
        (None, 0, False),
        ("Mobile", "mobile", False),
        ("1", 1, False),
    ],
)
def test_values_match(predicted, gold, expected) -> None:
    assert values_match(predicted, gold) is expected


def test_row_order_is_ignored_unless_asked() -> None:
    shuffled = [GOLD[2], GOLD[0], GOLD[1]]
    assert results_match(shuffled, GOLD) is True
    assert results_match(shuffled, GOLD, ordered=True) is False
    assert results_match(GOLD, GOLD, ordered=True) is True


def test_column_names_and_order_are_ignored_and_extra_columns_allowed() -> None:
    predicted = [(198, "Mobile", 49.5), (152, "Desktop", 38.0), (50, "Tablet", 12.5)]
    assert results_match(predicted, GOLD) is True


def test_missing_or_extra_rows_fail() -> None:
    assert results_match(GOLD[:2], GOLD) is False
    assert results_match([*GOLD, ("Other", 1)], GOLD) is False


def test_a_wrong_value_fails() -> None:
    assert results_match([("Mobile", 198), ("Desktop", 151), ("Tablet", 50)], GOLD) is False


def test_values_must_stay_together_in_their_row() -> None:
    swapped = [("Mobile", 152), ("Desktop", 198), ("Tablet", 50)]
    assert results_match(swapped, GOLD) is False


def test_fewer_columns_than_gold_fail() -> None:
    assert results_match([("Mobile",), ("Desktop",), ("Tablet",)], GOLD) is False


def test_empty_results() -> None:
    assert results_match([], []) is True
    assert results_match([("x",)], []) is False
    assert results_match([], GOLD) is False


def test_single_value_answer() -> None:
    assert results_match([(Decimal("59.05"),)], [(59.047619,)]) is True
    assert results_match([(105, 62, Decimal("59.05"))], [(59.05,)]) is True
