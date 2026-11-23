"""The analyst: question -> SQL plan -> guard -> run -> (self-correct) -> answer.

One question is handled in two LLM steps, without a framework:

1. Plan. The model sees the schema and replies in JSON mode with an action
   ("query", "clarify", "not_in_data", "refuse"), the SQL and a chart suggestion.
   The SQL goes through the guard (analyst/sql_guard.py) and runs on the read-only
   database. If the guard or DuckDB rejects it, the error message goes back to the
   model, which may fix it: at most `max_corrections` times.
2. Answer. The model sees the question, the SQL and the first rows of the result,
   and writes a short answer in the user's language. It has no tools: the worst a
   malicious comment in the data can do here is make the text wrong, which the
   evals check.

The chart itself is drawn by code (analyst/charts.py); the model only suggests the
chart type and columns, and the suggestion is dropped if the columns don't exist.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import duckdb

from analyst.database import DEFAULT_TIMEOUT_S, QueryTimeout, run_query
from analyst.llm_client import HistoryItem, LLMClient, LLMResponse
from analyst.normalize import Table
from analyst.pricing import estimate_cost_usd
from analyst.prompts import (
    ANSWER_SYSTEM_PROMPT,
    build_answer_message,
    build_schema_context,
    build_sql_system_prompt,
)
from analyst.sql_guard import DEFAULT_MAX_ROWS, SQLGuardError, check_sql
from analyst.tracing import NULL_TRACER, Tracer, trace_context

ACTIONS = {"query", "clarify", "not_in_data", "refuse"}
CHART_TYPES = {"bar", "line"}
JSON_MODE = {"type": "json_object"}
DEFAULT_MAX_CORRECTIONS = 2
PLAN_MAX_TOKENS = 1000
ANSWER_MAX_TOKENS = 500
# Error text sent back to the model is cut to this length.
MAX_ERROR_CHARS = 500

GIVE_UP_REPLY = (
    "Sorry, I could not write a working query for this question. "
    "Try rephrasing it or asking about one thing at a time."
)


class PlanError(ValueError):
    """The model's JSON reply is missing or malformed."""


@dataclass(frozen=True)
class ChartSpec:
    type: str  # "bar" | "line"
    x: str
    y: str


@dataclass(frozen=True)
class Plan:
    action: str
    sql: str
    assumptions: str
    message: str
    chart: dict[str, Any] | None


@dataclass
class SQLAttempt:
    sql: str  # as written by the model
    ok: bool
    ran_sql: str | None = None  # after the guard (normalized, LIMIT added)
    error: str | None = None
    error_type: str | None = None
    row_count: int | None = None
    duration_ms: float = 0.0


@dataclass
class AnalystResult:
    question: str
    conversation_id: str
    action: str = "error"  # one of ACTIONS, or "error" when we gave up
    answer: str = ""
    sql: str | None = None  # the SQL that ran and produced `rows`
    assumptions: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    truncated: bool = False  # the guard's LIMIT may have cut the result
    chart: ChartSpec | None = None
    attempts: list[SQLAttempt] = field(default_factory=list)
    self_corrections: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0
    latency_ms: float = 0.0  # time spent waiting for the LLM
    duration_ms: float = 0.0  # wall clock for the whole question


def parse_plan(text: str) -> Plan:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise PlanError(f"the reply is not valid JSON ({error.msg})") from error
    if not isinstance(data, dict):
        raise PlanError("the reply must be a JSON object")
    action = data.get("action")
    if action not in ACTIONS:
        raise PlanError(f"'action' must be one of {sorted(ACTIONS)}, got {action!r}")
    sql = data.get("sql") or ""
    if action == "query" and (not isinstance(sql, str) or not sql.strip()):
        raise PlanError("action 'query' needs a non-empty 'sql' string")
    chart = data.get("chart")
    return Plan(
        action=action,
        sql=sql.strip() if isinstance(sql, str) else "",
        assumptions=str(data.get("assumptions") or "").strip(),
        message=str(data.get("message") or "").strip(),
        chart=chart if isinstance(chart, dict) else None,
    )


def chart_for(plan_chart: dict[str, Any] | None, columns: list[str]) -> ChartSpec | None:
    """Keep the suggested chart only if it is a known type over existing columns."""
    if not plan_chart or len(columns) < 2:
        return None
    chart_type, x, y = plan_chart.get("type"), plan_chart.get("x"), plan_chart.get("y")
    if chart_type not in CHART_TYPES or x not in columns or y not in columns or x == y:
        return None
    return ChartSpec(type=chart_type, x=x, y=y)


class Analyst:
    def __init__(
        self,
        llm: LLMClient,
        connection: duckdb.DuckDBPyConnection,
        tables: list[Table],
        *,
        tracer: Tracer = NULL_TRACER,
        clock: Callable[[], datetime] = datetime.now,
        max_corrections: int = DEFAULT_MAX_CORRECTIONS,
        max_rows: int = DEFAULT_MAX_ROWS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.llm = llm
        self.connection = connection
        self.table_names = [table.name for table in tables]
        self.schema_context = build_schema_context(connection, tables)
        self.tracer = tracer
        self._clock = clock
        self.max_corrections = max_corrections
        self.max_rows = max_rows
        self.timeout_s = timeout_s

    # ------------------------------------------------------------------ public

    def today(self) -> datetime:
        """The date relative questions ("last month") are answered against."""
        return self._clock()

    def ask(self, question: str, *, conversation_id: str | None = None) -> AnalystResult:
        result = AnalystResult(
            question=question, conversation_id=conversation_id or uuid.uuid4().hex
        )
        started = time.perf_counter()
        with trace_context(conversation_id=result.conversation_id, turn_index=0):
            try:
                self._run(result)
            except Exception as error:
                result.duration_ms = (time.perf_counter() - started) * 1000
                self.tracer.emit(
                    "agent_turn",
                    ok=False,
                    error=error.__class__.__name__,
                    duration_ms=round(result.duration_ms, 1),
                )
                raise
            result.duration_ms = (time.perf_counter() - started) * 1000
            self._emit_turn(result)
        return result

    # ------------------------------------------------------------------ steps

    def _run(self, result: AnalystResult) -> None:
        system = build_sql_system_prompt(self.schema_context, self.today())
        history: list[HistoryItem] = [self.llm.user_message(result.question)]
        plan: Plan | None = None

        for attempt in range(self.max_corrections + 1):
            if attempt:
                result.self_corrections += 1
            response = self._chat_json(result, history, system)
            history.extend(response.history_items)
            try:
                plan = parse_plan(response.text)
            except PlanError as error:
                history.append(self._feedback(f"Your reply was not usable: {error}."))
                continue

            if plan.action != "query":
                result.action = plan.action
                result.answer = plan.message or plan.assumptions
                return

            sql_attempt = self._execute(plan.sql, result)
            result.attempts.append(sql_attempt)
            if sql_attempt.ok:
                break
            history.append(self._feedback(f"The SQL failed: {sql_attempt.error}"))
        else:
            result.action = "error"
            result.answer = GIVE_UP_REPLY
            return

        assert plan is not None
        result.action = "query"
        result.assumptions = plan.assumptions
        result.chart = chart_for(plan.chart, result.columns)
        result.answer = self._answer(result)

    def _execute(self, sql: str, result: AnalystResult) -> SQLAttempt:
        started = time.perf_counter()
        attempt = SQLAttempt(sql=sql, ok=False)
        try:
            guarded = check_sql(sql, self.table_names, max_rows=self.max_rows, today=self.today())
            attempt.ran_sql = guarded.sql
            query = run_query(self.connection, guarded.sql, timeout_s=self.timeout_s)
        except (SQLGuardError, QueryTimeout, duckdb.Error) as error:
            attempt.error_type = error.__class__.__name__
            attempt.error = str(error)[:MAX_ERROR_CHARS]
        else:
            attempt.ok = True
            attempt.row_count = len(query.rows)
            result.sql = guarded.sql
            result.columns = query.columns
            result.rows = query.rows
            result.truncated = guarded.limit_added and len(query.rows) >= self.max_rows
        attempt.duration_ms = (time.perf_counter() - started) * 1000
        self._emit_sql(attempt)
        return attempt

    def _answer(self, result: AnalystResult) -> str:
        message = build_answer_message(
            result.question,
            result.sql or "",
            result.assumptions,
            result.columns,
            result.rows,
            result.truncated,
        )
        response = self.llm.chat(
            [self.llm.user_message(message)],
            system=ANSWER_SYSTEM_PROMPT,
            max_tokens=ANSWER_MAX_TOKENS,
        )
        self._count(result, response)
        return response.text.strip()

    # ------------------------------------------------------------------ helpers

    def _chat_json(
        self, result: AnalystResult, history: list[HistoryItem], system: str
    ) -> LLMResponse:
        response = self.llm.chat(
            history, system=system, response_format=JSON_MODE, max_tokens=PLAN_MAX_TOKENS
        )
        self._count(result, response)
        # DeepSeek's JSON mode can return empty content; its docs say to retry.
        if not response.text.strip():
            response = self.llm.chat(
                history, system=system, response_format=JSON_MODE, max_tokens=PLAN_MAX_TOKENS
            )
            self._count(result, response)
        return response

    def _feedback(self, problem: str) -> HistoryItem:
        return self.llm.user_message(
            f"{problem}\nFix it and reply again with the same JSON object format."
        )

    @staticmethod
    def _count(result: AnalystResult, response: LLMResponse) -> None:
        result.llm_calls += 1
        result.input_tokens += response.input_tokens
        result.cached_input_tokens += response.cached_input_tokens
        result.output_tokens += response.output_tokens
        result.latency_ms += response.latency_ms
        cost = estimate_cost_usd(
            response.model,
            response.input_tokens,
            response.cached_input_tokens,
            response.output_tokens,
        )
        result.cost_usd = (
            None if cost is None or result.cost_usd is None else result.cost_usd + cost
        )

    def _emit_sql(self, attempt: SQLAttempt) -> None:
        fields: dict[str, Any] = {
            "name": "run_sql",
            "ok": attempt.ok,
            "duration_ms": round(attempt.duration_ms, 1),
        }
        # Without content logging, only the error type: DuckDB messages can quote data.
        error = attempt.error if self.tracer.include_content else attempt.error_type
        if not attempt.ok:
            fields["error"] = attempt.error_type
        fields["result"] = {"ok": attempt.ok, "row_count": attempt.row_count, "error": error}
        if self.tracer.include_content:
            fields["arguments"] = {"sql": attempt.sql}
        self.tracer.emit("tool_call", **fields)

    def _emit_turn(self, result: AnalystResult) -> None:
        fields: dict[str, Any] = {
            "ok": result.action != "error",
            "action": result.action,
            "llm_calls": result.llm_calls,
            "input_tokens": result.input_tokens,
            "cached_input_tokens": result.cached_input_tokens,
            "output_tokens": result.output_tokens,
            "cost_usd": result.cost_usd,
            "latency_ms": round(result.latency_ms, 1),
            "duration_ms": round(result.duration_ms, 1),
            "tool_calls": ["run_sql" for _ in result.attempts],
            "self_corrections": result.self_corrections,
            "row_count": len(result.rows),
        }
        if self.tracer.include_content:
            fields["user_text"] = result.question
            fields["reply"] = result.answer
            fields["sql"] = result.sql
        self.tracer.emit("agent_turn", **fields)
