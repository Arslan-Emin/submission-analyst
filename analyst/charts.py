"""Draw the chart for a result. The model only suggested type and columns (ChartSpec);
the drawing is plain code, so a chart can never show anything that is not in the rows.
"""

from __future__ import annotations

from datetime import date, datetime
from numbers import Number
from typing import Any

import altair as alt
import pandas as pd

from analyst.agent import ChartSpec

MAX_BARS = 30


def result_frame(columns: list[str], rows: list[tuple[Any, ...]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows, columns=columns)


def _is_numeric(values: list[Any]) -> bool:
    present = [v for v in values if v is not None]
    return bool(present) and all(isinstance(v, Number) and not isinstance(v, bool) for v in present)


def _is_temporal(values: list[Any]) -> bool:
    present = [v for v in values if v is not None]
    return bool(present) and all(isinstance(v, (date, datetime)) for v in present)


def build_chart(
    spec: ChartSpec | None, columns: list[str], rows: list[tuple[Any, ...]]
) -> alt.Chart | None:
    """An Altair chart, or None when the data does not fit the suggestion."""
    if spec is None or not rows:
        return None
    x_values = [row[columns.index(spec.x)] for row in rows]
    y_values = [row[columns.index(spec.y)] for row in rows]
    if not _is_numeric(y_values):
        return None
    frame = result_frame(columns, rows)[[spec.x, spec.y]]
    # DuckDB returns DECIMAL (e.g. from ROUND(100.0 * a / b, 2)) as Python Decimal,
    # which the chart's JSON encoder cannot serialize.
    frame = frame.assign(**{spec.y: frame[spec.y].astype(float)})
    # The y axis always starts at zero, so small differences don't look like cliffs.
    y = alt.Y(f"{spec.y}:Q", title=spec.y, scale=alt.Scale(zero=True))
    tooltip = [alt.Tooltip(spec.x), alt.Tooltip(spec.y)]

    if spec.type == "line":
        x_type = "T" if _is_temporal(x_values) else "O"
        x = alt.X(f"{spec.x}:{x_type}", title=spec.x)
        return alt.Chart(frame).mark_line(point=True).encode(x=x, y=y, tooltip=tooltip)

    if len(rows) > MAX_BARS:
        return None
    # Bars keep the order of the SQL result (e.g. ORDER BY count DESC).
    order = [str(v) for v in x_values]
    frame = frame.assign(**{spec.x: frame[spec.x].astype(str)})
    x = alt.X(f"{spec.x}:N", title=spec.x, sort=order)
    return alt.Chart(frame).mark_bar().encode(x=x, y=y, tooltip=tooltip)
