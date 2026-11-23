# Copied from conversational-form-agent (commit 6e67866), tests/test_agent.py:
# the scripted fake LLM and its reply helpers, moved to their own module.
"""Test doubles: a scripted fake LLM, so tests never call a real provider."""

from __future__ import annotations

from typing import Any

from analyst.llm_client import HistoryItem, LLMResponse, ToolCall, ToolSpec


def text_reply(text: str) -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=[],
        model="fake",
        input_tokens=10,
        output_tokens=5,
        latency_ms=1.0,
        history_items=[{"role": "assistant", "content": text}],
    )


def tool_reply(name: str, arguments: dict[str, Any] | None, call_id: str = "c1") -> LLMResponse:
    call = ToolCall(id=call_id, name=name, arguments=arguments, raw_arguments="...")
    return LLMResponse(
        text="",
        tool_calls=[call],
        model="fake",
        input_tokens=10,
        output_tokens=5,
        latency_ms=1.0,
        history_items=[{"role": "assistant", "tool_calls": [call_id]}],
    )


class ScriptedLLM:
    """Returns pre-written responses in order and records what it was sent."""

    model = "fake"

    def __init__(self, responses: list[LLMResponse | Exception]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []

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
    ) -> LLMResponse:
        self.requests.append({"history": list(history), "system": system, "tools": tools})
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response
