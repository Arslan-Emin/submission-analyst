"""Ask the analyst a question from the terminal (fixture data, real LLM).

python -m analyst.cli "Which complaint topic was most common last month?"
python -m analyst.cli --today 2026-11-23 "..."     # pin "today" (as the evals do)
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from analyst.agent import Analyst
from analyst.config import get_settings
from analyst.database import build_database, open_read_only
from analyst.factory import create_traced_llm, create_tracer
from analyst.normalize import normalize
from analyst.sources.fixture import FixtureSource

MAX_PRINTED_ROWS = 20


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask a question about the survey submissions.")
    parser.add_argument("question")
    parser.add_argument("--today", help="pin today's date, YYYY-MM-DD (default: real date)")
    args = parser.parse_args(argv)

    settings = get_settings()
    tracer = create_tracer(settings)
    llm = create_traced_llm(settings, tracer)
    today = datetime.strptime(args.today, "%Y-%m-%d").replace(hour=12) if args.today else None

    tables = normalize(FixtureSource().load())
    with tempfile.TemporaryDirectory() as folder:
        connection = open_read_only(build_database(tables, Path(folder) / "analyst.duckdb"))
        try:
            analyst = Analyst(
                llm,
                connection,
                tables,
                tracer=tracer,
                clock=(lambda: today) if today else datetime.now,
            )
            result = analyst.ask(args.question)
        finally:
            connection.close()

    print(f"[{result.action}] {result.answer}\n")
    if result.assumptions:
        print(f"Assumptions: {result.assumptions}\n")
    for attempt in result.attempts:
        status = "ok" if attempt.ok else f"failed: {attempt.error}"
        print(f"SQL ({status}):\n  {attempt.ran_sql or attempt.sql}\n")
    if result.columns:
        print(" | ".join(result.columns))
        for row in result.rows[:MAX_PRINTED_ROWS]:
            print(" | ".join("" if v is None else str(v) for v in row))
        if len(result.rows) > MAX_PRINTED_ROWS:
            print(f"... {len(result.rows) - MAX_PRINTED_ROWS} more rows")
        print()
    if result.chart:
        print(f"Chart: {result.chart.type} x={result.chart.x} y={result.chart.y}")
    cost = f"${result.cost_usd:.5f}" if result.cost_usd is not None else "unknown"
    print(
        f"LLM calls: {result.llm_calls}, tokens in/cached/out: {result.input_tokens}/"
        f"{result.cached_input_tokens}/{result.output_tokens}, cost: {cost}, "
        f"time: {result.duration_ms / 1000:.1f} s, self-corrections: {result.self_corrections}"
    )
    return 0 if result.action != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
