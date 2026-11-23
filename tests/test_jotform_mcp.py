"""MCP source tests with a fake tool session: no network, no OAuth."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from mcp.shared.auth import OAuthToken

from analyst.normalize import normalize
from analyst.sources.fixture import FixtureSource
from analyst.sources.jotform_mcp import (
    FileTokenStorage,
    JotformMCPSource,
    MCPError,
    call_read_tool,
    load_from_session,
    questions_from_answers,
)


class FakeSession:
    """Plays the MCP server: fetch returns the form, get_submissions pages the fixture."""

    def __init__(self, submissions: list[dict[str, Any]], *, structured: bool = True) -> None:
        self.submissions = submissions
        self.structured = structured
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        self.calls.append((name, arguments or {}))
        if name == "fetch":
            payload: Any = {"id": arguments["id"], "title": "Survey"}
        else:
            f = arguments["filter"]
            page = self.submissions[f["offset"] : f["offset"] + f["limit"]]
            payload = {"total_count": len(self.submissions), "submissions": page}
        if self.structured:
            return SimpleNamespace(isError=False, structuredContent=payload, content=[])
        text = SimpleNamespace(text=json.dumps(payload))
        return SimpleNamespace(isError=False, structuredContent=None, content=[text])


FIXTURE = FixtureSource().load()


def test_load_pages_submissions_and_rebuilds_the_questions() -> None:
    session = FakeSession(FIXTURE.submissions)

    data = asyncio.run(load_from_session(session, "f1"))

    assert data.form_title == "Survey"
    assert len(data.submissions) == len(FIXTURE.submissions)
    assert [name for name, _ in session.calls].count("get_submissions") == 4  # 100 per page
    # Same tables as the fixture, built from the questions recovered from answers.
    assert [[c.name for c in t.columns] for t in normalize(data)] == [
        [c.name for c in t.columns] for t in normalize(FIXTURE)
    ]


def test_text_content_results_are_parsed_too() -> None:
    data = asyncio.run(
        load_from_session(FakeSession(FIXTURE.submissions[:3], structured=False), "f1")
    )
    assert len(data.submissions) == 3


def test_questions_include_unanswered_ones_without_answers() -> None:
    questions = questions_from_answers(FIXTURE.submissions[:1])

    assert set(questions) == set(FIXTURE.questions)
    assert all("answer" not in q for q in questions.values())


@pytest.mark.parametrize("tool", ["create_form", "edit_form", "assign_form", "analyze_submissions"])
def test_only_read_tools_can_be_called(tool: str) -> None:
    session = FakeSession([])
    with pytest.raises(MCPError, match="not allowed"):
        asyncio.run(call_read_tool(session, tool, {}))
    assert session.calls == []  # refused before any request


def test_tool_errors_are_raised() -> None:
    class Failing:
        async def call_tool(self, name, arguments=None):
            return SimpleNamespace(isError=True, content=[SimpleNamespace(text="form not found")])

    with pytest.raises(MCPError, match="form not found"):
        asyncio.run(call_read_tool(Failing(), "fetch", {"id": "x"}))


def test_token_storage_round_trip(tmp_path) -> None:
    storage = FileTokenStorage(tmp_path / "tokens" / "jotform.json")
    assert asyncio.run(storage.get_tokens()) is None

    token = OAuthToken(access_token="a", token_type="Bearer", refresh_token="r")
    asyncio.run(storage.set_tokens(token))

    assert asyncio.run(storage.get_tokens()).refresh_token == "r"


def test_source_uses_the_injected_session_runner() -> None:
    async def runner(action):
        return await action(FakeSession(FIXTURE.submissions[:5]))

    data = JotformMCPSource("f1", runner=runner).load()

    assert data.form_id == "f1" and len(data.submissions) == 5
