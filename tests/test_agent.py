"""Analyst loop tests: scripted fake LLM, real DuckDB with the fixture data."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from analyst.agent import GIVE_UP_REPLY, Analyst, PlanError, chart_for, parse_plan
from analyst.database import build_database, open_read_only
from analyst.normalize import normalize
from analyst.sources.fixture import FixtureSource
from analyst.tracing import InMemorySink, TracedLLMClient, Tracer
from tests.fakes import ScriptedLLM, text_reply

FIXED_NOW = datetime(2026, 11, 23, 12, 0)
COUNT_SQL = "SELECT device, count(*) AS n FROM submissions GROUP BY device ORDER BY n DESC"


def plan(action: str = "query", sql: str = COUNT_SQL, **extra) -> str:
    body = {"action": action, "sql": sql, "assumptions": "", "message": "", **extra}
    body.setdefault("chart", {"type": "bar", "x": "device", "y": "n"})
    return json.dumps(body)


@pytest.fixture(scope="module")
def database(tmp_path_factory):
    tables = normalize(FixtureSource().load())
    path = build_database(tables, tmp_path_factory.mktemp("db") / "f.duckdb")
    connection = open_read_only(path)
    yield connection, tables
    connection.close()


def make(database, replies, *, traced: bool = False, include_content: bool = False):
    connection, tables = database
    fake = ScriptedLLM([r if isinstance(r, Exception) else text_reply(r) for r in replies])
    sink = InMemorySink()
    tracer = Tracer([sink], include_content=include_content)
    llm = TracedLLMClient(fake, tracer, provider="fake") if traced else fake
    analyst = Analyst(llm, connection, tables, tracer=tracer, clock=lambda: FIXED_NOW)
    return analyst, fake, sink


def test_question_is_answered_with_sql_rows_chart_and_answer(database) -> None:
    analyst, fake, _ = make(database, [plan(), "Mobile leads with 198 submissions."])

    result = analyst.ask("How many submissions per device?")

    assert result.action == "query"
    assert result.answer == "Mobile leads with 198 submissions."
    assert result.columns == ["device", "n"]
    assert result.rows[0] == ("Mobile", 198)
    assert result.sql.endswith("LIMIT 1000")
    assert result.chart is not None and (result.chart.x, result.chart.y) == ("device", "n")
    assert (result.llm_calls, result.self_corrections) == (2, 0)
    # Plan step in JSON mode with the pinned date; answer step plain, with the rows as data.
    plan_request, answer_request = fake.requests
    assert plan_request["response_format"] == {"type": "json_object"}
    assert "Today is 2026-11-23" in plan_request["system"]
    assert "'Mobile', 'Desktop', 'Tablet'" in plan_request["system"]
    assert answer_request["response_format"] is None
    assert '["Mobile", 198]' in answer_request["history"][0]["content"]


def test_sql_error_is_sent_back_and_the_model_fixes_it(database) -> None:
    bad = "SELECT devise, count(*) AS n FROM submissions GROUP BY devise"
    analyst, fake, _ = make(database, [plan(sql=bad), plan(), "ok"])

    result = analyst.ask("Submissions per device?")

    assert result.action == "query"
    assert result.self_corrections == 1
    assert [a.ok for a in result.attempts] == [False, True]
    assert result.attempts[0].error_type == "BinderException"
    feedback = fake.requests[1]["history"][-1]["content"]
    assert "The SQL failed" in feedback and "devise" in feedback


def test_guard_rejection_is_reported_and_nothing_destructive_runs(database) -> None:
    refusal = plan(action="refuse", sql="", message="I can only read the data.")
    analyst, fake, _ = make(database, [plan(sql="DROP TABLE submissions"), refusal])

    result = analyst.ask("Delete the table")

    assert result.action == "refuse"
    assert result.answer == "I can only read the data."
    assert result.attempts[0].error_type == "SQLGuardError"
    assert "Only read-only SELECT" in fake.requests[1]["history"][-1]["content"]
    assert result.llm_calls == 2  # no answer step for a refusal


def test_gives_up_after_the_correction_budget(database) -> None:
    bad = plan(sql="SELECT nope FROM submissions")
    analyst, fake, _ = make(database, [bad, bad, bad])

    result = analyst.ask("?")

    assert result.action == "error"
    assert result.answer == GIVE_UP_REPLY
    assert (len(result.attempts), result.self_corrections, result.llm_calls) == (3, 2, 3)
    assert result.rows == []


def test_invalid_json_is_corrected(database) -> None:
    analyst, fake, _ = make(database, ["SELECT * FROM submissions", plan(), "ok"])

    result = analyst.ask("Per device?")

    assert result.action == "query"
    assert result.self_corrections == 1
    assert "not valid JSON" in fake.requests[1]["history"][-1]["content"]


def test_empty_json_mode_reply_is_retried_once_without_feedback(database) -> None:
    analyst, fake, _ = make(database, ["", plan(), "ok"])

    result = analyst.ask("Per device?")

    assert result.action == "query"
    assert (result.llm_calls, result.self_corrections) == (3, 0)
    assert fake.requests[1]["history"] == fake.requests[0]["history"]


@pytest.mark.parametrize("action", ["clarify", "not_in_data"])
def test_non_query_actions_return_the_message_without_running_sql(database, action) -> None:
    analyst, _, _ = make(database, [plan(action=action, sql="", message="Which period?")])

    result = analyst.ask("Who are the best customers?")

    assert (result.action, result.answer) == (action, "Which period?")
    assert result.attempts == [] and result.llm_calls == 1


def test_events_for_one_question_share_a_conversation_id(database) -> None:
    bad = plan(sql="SELECT nope FROM submissions")
    analyst, _, sink = make(database, [bad, plan(), "ok"], traced=True)

    result = analyst.ask("Per device?")

    types = [e["type"] for e in sink.events]
    assert types == ["llm_call", "tool_call", "llm_call", "tool_call", "llm_call", "agent_turn"]
    assert {e["conversation_id"] for e in sink.events} == {result.conversation_id}
    assert {e["turn_index"] for e in sink.events} == {0}
    failed, succeeded = (e for e in sink.events if e["type"] == "tool_call")
    assert failed["name"] == "run_sql" and failed["ok"] is False
    assert failed["result"] == {"ok": False, "row_count": None, "error": "BinderException"}
    assert succeeded["result"]["row_count"] == 3
    turn = sink.events[-1]
    assert (turn["llm_calls"], turn["self_corrections"], turn["tool_calls"]) == (
        3,
        1,
        ["run_sql", "run_sql"],
    )


def test_sql_and_text_are_traced_only_with_content_logging(database) -> None:
    analyst, _, plain = make(database, [plan(), "Mobile leads."], traced=True)
    analyst.ask("Per device?")
    assert "GROUP BY" not in json.dumps(plain.events)
    assert "Mobile leads" not in json.dumps(plain.events)

    analyst, _, full = make(database, [plan(), "Mobile leads."], include_content=True)
    analyst.ask("Per device?")
    turn = full.events[-1]
    assert turn["reply"] == "Mobile leads." and "GROUP BY" in turn["sql"]
    assert "GROUP BY" in full.events[0]["arguments"]["sql"]


def test_llm_failure_is_traced_and_raised(database) -> None:
    analyst, _, sink = make(database, [RuntimeError("provider down")])

    with pytest.raises(RuntimeError):
        analyst.ask("?")

    assert sink.events[-1]["type"] == "agent_turn"
    assert sink.events[-1]["ok"] is False


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("[]", "JSON object"),
        ('{"action": "delete"}', "'action' must be one of"),
        ('{"action": "query", "sql": ""}', "non-empty 'sql'"),
    ],
)
def test_parse_plan_rejects_bad_replies(text: str, problem: str) -> None:
    with pytest.raises(PlanError, match=problem):
        parse_plan(text)


@pytest.mark.parametrize(
    ("chart", "expected"),
    [
        ({"type": "line", "x": "week", "y": "avg"}, ("line", "week", "avg")),
        ({"type": "pie", "x": "week", "y": "avg"}, None),
        ({"type": "bar", "x": "week", "y": "missing"}, None),
        ({"type": "bar", "x": "week", "y": "week"}, None),
        (None, None),
    ],
)
def test_chart_suggestion_is_validated_against_result_columns(chart, expected) -> None:
    spec = chart_for(chart, ["week", "avg"])
    assert (spec and (spec.type, spec.x, spec.y)) == expected


def test_generated_sql_uses_the_analyst_clock_not_the_system_clock(database) -> None:
    sql = "SELECT CURRENT_DATE AS today"
    analyst, _, _ = make(database, [plan(sql=sql, chart={"type": "none"}), "ok"])

    result = analyst.ask("What day is it?")

    assert str(result.rows[0][0]) == "2026-11-23"
    assert "CAST('2026-11-23' AS DATE)" in result.sql
