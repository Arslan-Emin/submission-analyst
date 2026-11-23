"""CLI smoke test with a scripted LLM: builds the fixture database and prints the result."""

from __future__ import annotations

import json

from analyst import cli
from analyst.tracing import Tracer
from tests.fakes import ScriptedLLM, text_reply


def test_cli_prints_answer_sql_and_rows(monkeypatch, capsys) -> None:
    plan = {
        "action": "query",
        "sql": "SELECT count(*) AS n FROM submissions",
        "assumptions": "",
        "message": "",
        "chart": {"type": "none"},
    }
    fake = ScriptedLLM([text_reply(json.dumps(plan)), text_reply("There are 400 submissions.")])
    monkeypatch.setattr(cli, "create_tracer", lambda settings: Tracer())
    monkeypatch.setattr(cli, "create_traced_llm", lambda settings, tracer: fake)

    exit_code = cli.main(["How many submissions?"])  # fixture snapshot: 2026-11-23

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "[query] There are 400 submissions." in out
    assert "SELECT COUNT(*) AS n FROM submissions LIMIT 1000" in out  # as normalized by the guard
    assert "\n400\n" in out
    assert "Today is 2026-11-23" in fake.requests[0]["system"]


def test_today_can_be_overridden(monkeypatch) -> None:
    plan = {"action": "clarify", "sql": "", "message": "Which period?"}
    fake = ScriptedLLM([text_reply(json.dumps(plan))])
    monkeypatch.setattr(cli, "create_tracer", lambda settings: Tracer())
    monkeypatch.setattr(cli, "create_traced_llm", lambda settings, tracer: fake)

    cli.main(["--today", "2026-12-15", "Last month?"])

    assert "Today is 2026-12-15" in fake.requests[0]["system"]
