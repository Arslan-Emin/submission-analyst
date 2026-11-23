"""Eval set and runner tests: the gold queries are sound, and a model that writes
the gold SQL (or picks the expected action) passes every case."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from analyst.agent import AnalystResult
from analyst.llm_client import HistoryItem, LLMResponse, ToolSpec
from analyst.sql_guard import check_sql
from analyst.tracing import Tracer
from evals import run_evals
from tests.fakes import text_reply

CASES = run_evals.load_cases()
CATEGORIES = {
    "filter_count",
    "group_by",
    "date_range",
    "ratio",
    "join",
    "ambiguous",
    "not_in_data",
    "harmful",
    "prompt_injection",
}


class OracleLLM:
    """Answers each eval question with its gold SQL or expected action."""

    model = "fake"

    def __init__(self, cases: list[dict[str, Any]], wrong: set[str] = frozenset()) -> None:
        self.by_question = {case["question"]: case for case in cases}
        self.wrong = wrong

    def user_message(self, text: str) -> HistoryItem:
        return {"role": "user", "content": text}

    def assistant_message(self, text: str) -> HistoryItem:
        return {"role": "assistant", "content": text}

    def tool_result(self, call_id: str, output: str) -> HistoryItem:
        return {"role": "tool", "tool_call_id": call_id, "content": output}

    def chat(
        self,
        history: list[HistoryItem],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if response_format is None:  # answer step
            return text_reply("I need a dark mode. Done.")
        case = self.by_question[history[0]["content"]]
        if case["id"] in self.wrong:
            plan = {"action": "query", "sql": "SELECT 1 AS x FROM submissions LIMIT 1"}
        elif "gold_sql" in case:
            plan = {"action": "query", "sql": case["gold_sql"]}
        elif case["expect"] == "clarify_or_assume":
            plan = {"action": "clarify", "sql": "", "message": "Which measure?"}
        else:
            plan = {"action": case["expect"], "sql": "", "message": "No."}
        return text_reply(json.dumps(plan))


def test_eval_set_shape() -> None:
    ids = [case["id"] for case in CASES]
    assert 25 <= len(CASES) <= 30
    assert len(ids) == len(set(ids))
    assert {case["category"] for case in CASES} == CATEGORIES
    for case in CASES:
        assert ("gold_sql" in case) != ("expect" in case), case["id"]
        if "expect" in case:
            assert case["expect"] in run_evals.ACCEPTED_ACTIONS


@pytest.fixture(scope="module")
def analyst(tmp_path_factory):
    from analyst.factory import create_analyst
    from analyst.sources.fixture import FixtureSource

    built = create_analyst(
        FixtureSource(),
        tmp_path_factory.mktemp("db") / "e.duckdb",
        OracleLLM(CASES),
        Tracer(),
    )
    yield built
    built.connection.close()


@pytest.mark.parametrize("case", [c for c in CASES if "gold_sql" in c], ids=lambda c: c["id"])
def test_gold_sql_passes_the_guard_and_returns_rows(analyst, case) -> None:
    check_sql(case["gold_sql"], analyst.table_names)
    assert run_evals.gold_rows(analyst, case)


def test_oracle_model_passes_every_case(analyst) -> None:
    records = run_evals.run(CASES, analyst, repeats=1, progress=False)

    failed = {r["id"]: r["checks"] for r in records if not r["passed"]}
    assert failed == {}
    summary = run_evals.summarize(records)
    assert summary["cases"]["rate"] == 1.0
    assert summary["execution_accuracy"]["total"] == sum("gold_sql" in c for c in CASES)


def test_wrong_rows_wrong_action_and_guard_rejections_fail(analyst) -> None:
    case = next(c for c in CASES if c["id"] == "fc_low_scores")
    refuse = next(c for c in CASES if c["id"] == "harm_drop")
    analyst.llm = OracleLLM(CASES, wrong={"fc_low_scores", "harm_drop"})
    try:
        low, drop = run_evals.run([case, refuse], analyst, repeats=1, progress=False)
    finally:
        analyst.llm = OracleLLM(CASES)

    assert low["passed"] is False
    assert {c["kind"]: c["passed"] for c in low["checks"]}["execution"] is False
    assert drop["passed"] is False  # ran a query instead of refusing


def test_assumption_is_required_when_an_ambiguous_question_is_answered(analyst) -> None:
    case = next(c for c in CASES if c["id"] == "amb_best_customers")
    result = AnalystResult(question=case["question"], conversation_id="x", action="query")
    assert run_evals.score(case, result, None)[0]["passed"] is False
    result.assumptions = "Best = highest NPS."
    assert run_evals.score(case, result, None)[0]["passed"] is True


def test_main_writes_a_results_file_evalplat_can_read(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(run_evals, "create_tracer", lambda *a, **k: Tracer())
    monkeypatch.setattr(run_evals, "create_traced_llm", lambda s, t: OracleLLM(CASES))

    code = run_evals.main(
        ["--cases", "fc_low_scores,harm_drop", "--repeats", "2", "--out-dir", str(tmp_path)]
    )

    assert code == 0
    (out,) = tmp_path.glob("*_fake.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["run"]["model"] == "fake" and payload["run"]["run_at_utc"]
    assert payload["run"]["eval_clock"].startswith("2026-11-23")
    assert payload["run"]["repeats"] == 2
    assert [c["id"] for c in payload["cases"]] == ["fc_low_scores", "harm_drop"] * 2
    first = payload["cases"][0]
    assert first["passed"] is True and first["category"] == "filter_count"
    assert len(first["conversation_id"]) == 32
    assert first["transcript"][0]["tool_calls"][0]["name"] == "run_sql"
    assert payload["summary"]["cases"] == {"passed": 4, "total": 4, "rate": 1.0}


def test_unknown_case_ids_are_rejected(tmp_path: Path) -> None:
    assert run_evals.main(["--cases", "nope", "--out-dir", str(tmp_path)]) == 2
