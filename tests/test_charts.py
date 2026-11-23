"""Chart tests: the spec turns into the right Altair chart, or into no chart."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from analyst.agent import ChartSpec
from analyst.charts import build_chart

DEVICES = (["device", "n"], [("Mobile", 198), ("Desktop", 152), ("Tablet", 50)])


def test_bar_chart_keeps_result_order_and_zero_based_axis() -> None:
    chart = build_chart(ChartSpec("bar", "device", "n"), *DEVICES)

    spec = chart.to_dict()
    assert spec["mark"]["type"] == "bar"
    assert spec["encoding"]["x"]["sort"] == ["Mobile", "Desktop", "Tablet"]
    assert spec["encoding"]["y"]["scale"]["zero"] is True


def test_line_chart_uses_a_time_axis_for_dates() -> None:
    rows = [(datetime(2026, 10, 5), 3.4), (datetime(2026, 10, 12), 3.1)]

    spec = build_chart(
        ChartSpec("line", "week", "avg_score"), ["week", "avg_score"], rows
    ).to_dict()

    assert spec["mark"]["type"] == "line"
    assert spec["encoding"]["x"]["type"] == "temporal"


def test_no_chart_for_text_values_empty_results_or_too_many_bars() -> None:
    assert build_chart(ChartSpec("bar", "n", "device"), *DEVICES) is None
    assert build_chart(ChartSpec("bar", "device", "n"), ["device", "n"], []) is None
    many = [(str(i), i) for i in range(31)]
    assert build_chart(ChartSpec("bar", "device", "n"), ["device", "n"], many) is None
    assert build_chart(None, *DEVICES) is None


def test_nulls_do_not_break_the_chart() -> None:
    rows = [("Istanbul", 10), (None, 4)]

    assert build_chart(ChartSpec("bar", "city", "n"), ["city", "n"], rows) is not None


def test_decimal_values_from_duckdb_can_be_serialized() -> None:
    # ROUND(100.0 * a / b, 2) comes back from DuckDB as Decimal.
    rows = [("Mobile", Decimal("59.05")), ("Desktop", Decimal("28.57"))]

    chart = build_chart(ChartSpec("bar", "device", "pct"), ["device", "pct"], rows)

    assert '"pct": 59.05' in chart.to_json()
