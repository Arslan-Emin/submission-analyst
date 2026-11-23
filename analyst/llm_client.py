# Copied from conversational-form-agent (commit 6e67866), app/llm_client.py.
# Changes: imports point to analyst.config; the OpenAI Responses API client was dropped
# (this project only uses DeepSeek through the Chat Completions API).
"""Thin, provider-agnostic wrapper around LLM APIs.

Every LLM call in the project goes through this module. That gives us one place
to measure tokens and latency, and to add cost tracking and tracing
(analyst/tracing.py), without touching the agent code.

The agent only depends on the `LLMClient` protocol and the small dataclasses
below. Conversation history items are provider-specific dicts, but the agent
never builds them by hand: it asks the client for them (`user_message`,
`tool_result`, `LLMResponse.history_items`). Swapping providers therefore means
picking another class here, not rewriting the agent.

The implementation, `ChatCompletionsClient`, uses the OpenAI-compatible Chat
Completions API. It works with DeepSeek (our default) and any other
OpenAI-compatible provider via `base_url`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import OpenAI

from analyst.config import Settings, get_settings

HistoryItem = dict[str, Any]

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call. `parameters` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """A tool call requested by the model.

    `arguments` is None when the model produced invalid JSON; the agent can then
    report the error back to the model instead of crashing.
    """

    id: str
    name: str
    arguments: dict[str, Any] | None
    raw_arguments: str


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    # Part of input_tokens served from the provider's prompt cache (billed cheaper).
    cached_input_tokens: int = 0
    # Provider-format items to append to the conversation history as-is.
    history_items: list[HistoryItem] = field(default_factory=list)


class LLMClient(Protocol):
    model: str

    def user_message(self, text: str) -> HistoryItem: ...

    def assistant_message(self, text: str) -> HistoryItem: ...

    def tool_result(self, call_id: str, output: str) -> HistoryItem: ...

    def chat(
        self,
        history: list[HistoryItem],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse: ...


class ChatCompletionsClient:
    """`LLMClient` on top of the OpenAI-compatible Chat Completions API.

    The full history is sent on every request: conversation state lives in our
    code, which keeps eval runs reproducible.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        extra_body: dict[str, Any] | None = None,
        sdk_client: Any = None,
    ) -> None:
        self.model = model
        # Provider-specific request fields (e.g. DeepSeek's `thinking` switch).
        self._extra_body = extra_body or {}
        # `sdk_client` lets tests inject a fake instead of hitting the network.
        self._client = (
            sdk_client if sdk_client is not None else OpenAI(api_key=api_key, base_url=base_url)
        )

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
        # The system prompt is added per request, not stored in history, so it
        # can change (e.g. today's date) without rewriting the conversation.
        messages = ([{"role": "system", "content": system}] if system else []) + history
        request: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            request["tools"] = [_to_chat_tool(tool) for tool in tools]
        if self._extra_body:
            request["extra_body"] = self._extra_body

        start = time.perf_counter()
        response = self._client.chat.completions.create(**request)
        latency_ms = (time.perf_counter() - start) * 1000

        message = response.choices[0].message
        tool_calls = [
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=_parse_arguments(call.function.arguments),
                raw_arguments=call.function.arguments,
            )
            for call in (message.tool_calls or [])
        ]
        usage = response.usage
        details = usage.prompt_tokens_details if usage else None
        # model_dump keeps provider-specific extras such as DeepSeek's
        # `reasoning_content`, which DeepSeek requires to be sent back when tools are used.
        assistant_item = message.model_dump(mode="json", exclude_none=True)
        return LLMResponse(
            text=message.content or "",
            tool_calls=tool_calls,
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cached_input_tokens=(details.cached_tokens or 0) if details else 0,
            latency_ms=latency_ms,
            history_items=[assistant_item],
        )


def _to_chat_tool(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _parse_arguments(raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def create_llm_client(settings: Settings | None = None) -> LLMClient:
    """Build the configured LLM client. Fails fast with a clear message if misconfigured."""
    settings = settings or get_settings()

    if settings.llm_provider == "deepseek":
        if not settings.deepseek_api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        if settings.deepseek_thinking not in {"enabled", "disabled"}:
            raise RuntimeError("DEEPSEEK_THINKING must be 'enabled' or 'disabled'.")
        return ChatCompletionsClient(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=DEEPSEEK_BASE_URL,
            extra_body={"thinking": {"type": settings.deepseek_thinking}},
        )

    raise RuntimeError(f"Unknown LLM_PROVIDER {settings.llm_provider!r}. Use 'deepseek'.")
