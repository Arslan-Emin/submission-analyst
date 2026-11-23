"""Run the eval questions against the analyst and write a results file.

    python -m evals.run_evals                      # every case once
    python -m evals.run_evals --repeats 3          # LLMs are not deterministic
    python -m evals.run_evals --cases fc_low_scores,harm_drop

Data: the fixture only (offline, deterministic), with "today" pinned to its
snapshot date. The results file has the same format as the form agent's, so the
eval platform (`evalplat ingest / compare / gate`) reads both.

A case passes when every one of its checks passes:
- action      the analyst chose the expected action (query / refuse / ...)
- execution   query cases: the rows equal the gold query's rows (evals/scoring.py)
- sql_safety  the model never wrote SQL that the guard had to reject
- answer      some cases: the answer text must (not) contain given phrases
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from analyst.agent import Analyst, AnalystResult
from analyst.config import Settings, get_settings
from analyst.database import run_query
from analyst.factory import PROJECT_ROOT, create_analyst, create_traced_llm, create_tracer
from analyst.sources.fixture import FixtureSource
from evals.scoring import results_match

EVALS_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = EVALS_DIR / "questions.json"
RESULTS_DIR = EVALS_DIR / "results"

ACCEPTED_ACTIONS = {
    "refuse": {"refuse"},
    "not_in_data": {"not_in_data"},
    "clarify_or_assume": {"clarify", "query"},  # "query" only with a stated assumption
}


def load_cases(path: Path = QUESTIONS_FILE) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def gold_rows(analyst: Analyst, case: dict[str, Any]) -> list[tuple[Any, ...]] | None:
    if "gold_sql" not in case:
        return None
    return run_query(analyst.connection, case["gold_sql"]).rows


def _check(kind: str, description: str, passed: bool) -> dict[str, Any]:
    return {"kind": kind, "description": description, "passed": bool(passed), "turn": 0}


def score(
    case: dict[str, Any], result: AnalystResult, gold: list[tuple[Any, ...]] | None
) -> list[dict[str, Any]]:
    checks = []
    if gold is not None:
        checks.append(_check("action", "ran a query", result.action == "query"))
        checks.append(
            _check(
                "execution",
                "result matches gold"
                + (" (ordered)" if case.get("ordered") else "")
                + (f" (top {case['top_k']})" if case.get("top_k") else ""),
                result.action == "query"
                and results_match(
                    result.rows,
                    gold,
                    ordered=bool(case.get("ordered")),
                    top_k=case.get("top_k"),
                ),
            )
        )
    else:
        expect = case["expect"]
        passed = result.action in ACCEPTED_ACTIONS[expect]
        if expect == "clarify_or_assume" and result.action == "query":
            passed = bool(result.assumptions)
        checks.append(_check("action", f"expected {expect}, got {result.action}", passed))

    rejected = [a for a in result.attempts if a.error_type == "SQLGuardError"]
    checks.append(_check("sql_safety", "no SQL rejected by the guard", not rejected))

    answer = result.answer.casefold()
    if must := case.get("answer_must_contain_any"):
        checks.append(
            _check(
                "answer",
                f"answer mentions one of {must}",
                any(p.casefold() in answer for p in must),
            )
        )
    if forbidden := case.get("answer_must_not_contain"):
        checks.append(
            _check(
                "answer",
                f"answer contains none of {forbidden}",
                not any(p.casefold() in answer for p in forbidden),
            )
        )
    return checks


def run_case(
    analyst: Analyst, case: dict[str, Any], gold: list[tuple[Any, ...]] | None
) -> dict[str, Any]:
    conversation_id = uuid.uuid4().hex
    record: dict[str, Any] = {
        "id": case["id"],
        "category": case["category"],
        "conversation_id": conversation_id,
        "passed": False,
        "error": None,
        "expected_submit": None,
        "submitted": False,
        "checks": [],
        "final_values": {},
        "transcript": [],
        "metrics": {},
    }
    started = time.perf_counter()
    try:
        result = analyst.ask(case["question"], conversation_id=conversation_id)
    except Exception as error:  # noqa: BLE001 - a crash is a failed case, not a failed run
        record["error"] = f"{error.__class__.__name__}: {error}"[:300]
        record["metrics"] = {
            "user_turns": 1,
            "wall_time_s": round(time.perf_counter() - started, 2),
        }
        return record

    checks = score(case, result, gold)
    record["checks"] = checks
    record["passed"] = all(c["passed"] for c in checks)
    record["final_values"] = {
        "action": result.action,
        "assumptions": result.assumptions,
        "sql": result.sql,
        "row_count": len(result.rows),
        "rows_preview": [[_jsonable(v) for v in row] for row in result.rows[:10]],
    }
    record["transcript"] = [
        {
            "user": case["question"],
            "agent": result.answer,
            "tool_calls": [
                {
                    "name": "run_sql",
                    "arguments": {"sql": attempt.sql},
                    "ok": attempt.ok,
                    "error": attempt.error,
                }
                for attempt in result.attempts
            ],
            "hit_iteration_limit": result.action == "error",
        }
    ]
    record["metrics"] = {
        "user_turns": 1,
        "llm_calls": result.llm_calls,
        "input_tokens": result.input_tokens,
        "cached_input_tokens": result.cached_input_tokens,
        "output_tokens": result.output_tokens,
        "latency_ms": round(result.latency_ms, 1),
        "cost_usd": round(result.cost_usd, 6) if result.cost_usd is not None else None,
        "wall_time_s": round(time.perf_counter() - started, 2),
        "self_corrections": result.self_corrections,
    }
    return record


def _jsonable(value: Any) -> Any:
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _rate(passed: int, total: int) -> dict[str, Any]:
    return {"passed": passed, "total": total, "rate": round(passed / total, 3) if total else None}


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(1, math.ceil(pct / 100 * len(ordered))) - 1]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    def checks_of(kind: str) -> list[bool]:
        return [c["passed"] for r in records for c in r["checks"] if c["kind"] == kind]

    by_category: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        by_category[record["category"]].append(record["passed"])
    metrics = [r["metrics"] for r in records if "llm_calls" in r["metrics"]]
    costs = [m["cost_usd"] for m in metrics if m.get("cost_usd") is not None]
    walls = [m["wall_time_s"] for m in metrics]
    n = len(metrics) or 1
    execution, action, safety = checks_of("execution"), checks_of("action"), checks_of("sql_safety")
    return {
        "cases": _rate(sum(r["passed"] for r in records), len(records)),
        "errors": sum(r["error"] is not None for r in records),
        "execution_accuracy": _rate(sum(execution), len(execution)),
        "action_accuracy": _rate(sum(action), len(action)),
        "sql_safety": _rate(sum(safety), len(safety)),
        "answer_checks": _rate(sum(checks_of("answer")), len(checks_of("answer"))),
        "by_category": {k: _rate(sum(v), len(v)) for k, v in sorted(by_category.items())},
        "self_corrections_total": sum(m.get("self_corrections", 0) for m in metrics),
        "cases_with_self_correction": sum(m.get("self_corrections", 0) > 0 for m in metrics),
        "avg_llm_calls": round(sum(m["llm_calls"] for m in metrics) / n, 6),
        "avg_input_tokens": round(sum(m["input_tokens"] for m in metrics) / n, 6),
        "avg_output_tokens": round(sum(m["output_tokens"] for m in metrics) / n, 6),
        "avg_cost_usd": round(sum(costs) / len(costs), 6) if costs else None,
        "total_cost_usd": round(sum(costs), 6),
        "avg_turn_latency_s": round(sum(walls) / len(walls), 2) if walls else None,
        "p95_turn_latency_s": _percentile(walls, 95),
    }


def run(
    cases: list[dict[str, Any]],
    analyst: Analyst,
    *,
    repeats: int,
    progress: bool = True,
) -> list[dict[str, Any]]:
    golds = {case["id"]: gold_rows(analyst, case) for case in cases}
    records = []
    for repeat in range(repeats):
        for case in cases:
            record = run_case(analyst, case, golds[case["id"]])
            records.append(record)
            if progress:
                status = "PASS" if record["passed"] else ("ERROR" if record["error"] else "FAIL")
                failed = [c["kind"] for c in record["checks"] if not c["passed"]]
                print(
                    f"[{repeat + 1}/{repeats}] {status:5} {case['id']}"
                    + (f"  failed: {', '.join(failed)}" if failed else ""),
                    flush=True,
                )
    return records


def main(argv: list[str] | None = None, settings: Settings | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the analyst evals on the fixture.")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cases", help="comma-separated case ids (default: all)")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_FILE)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    cases = load_cases(args.questions)
    if args.cases:
        wanted = set(args.cases.split(","))
        unknown = wanted - {c["id"] for c in cases}
        if unknown:
            print(f"Unknown case ids: {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
        cases = [c for c in cases if c["id"] in wanted]

    settings = settings or get_settings()
    started_at = datetime.now(UTC).replace(microsecond=0)
    stamp = started_at.strftime("%Y%m%d-%H%M%S")
    trace_name = f"eval_{stamp}.jsonl"
    # Evals use fake data only, so the full content is traced.
    tracer = create_tracer(settings, file_name=trace_name, include_content=True)
    llm = create_traced_llm(settings, tracer)

    wall_start = time.perf_counter()
    with tempfile.TemporaryDirectory() as folder:
        analyst = create_analyst(FixtureSource(), Path(folder) / "eval.duckdb", llm, tracer)
        try:
            records = run(cases, analyst, repeats=args.repeats)
            eval_clock = analyst.today().isoformat()
        finally:
            analyst.connection.close()

    trace_dir = Path(settings.trace_dir)
    payload = {
        "run": {
            "run_at_utc": started_at.isoformat(),
            "provider": settings.llm_provider,
            "model": llm.model,
            "deepseek_thinking": settings.deepseek_thinking,
            "eval_clock": eval_clock,
            "cases_file": args.questions.name,
            "repeats": args.repeats,
            "trace_file": (trace_dir / trace_name).as_posix() if settings.trace_enabled else None,
            "wall_time_s": round(time.perf_counter() - wall_start, 1),
        },
        "summary": summarize(records),
        "cases": records,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{stamp}_{llm.model}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    summary = payload["summary"]
    print(
        f"\nPassed {summary['cases']['passed']}/{summary['cases']['total']} "
        f"(execution {summary['execution_accuracy']['rate']}, "
        f"actions {summary['action_accuracy']['rate']}), errors {summary['errors']}, "
        f"cost ${summary['total_cost_usd']:.4f}"
    )
    try:
        shown = out.relative_to(PROJECT_ROOT)
    except ValueError:
        shown = out
    print(f"Results: {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
