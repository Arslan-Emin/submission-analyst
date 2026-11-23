# Copied from conversational-form-agent (commit 6e67866), tests/test_llm_client.py.
# Changes: the OpenAI Responses API tests were dropped with that client.
"""Unit tests for the LLM clients. No network calls: the SDK is faked."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage, PromptTokensDetails

from analyst.config import Settings
from analyst.llm_client import (
    DEEPSEEK_BASE_URL,
    ChatCompletionsClient,
    ToolSpec,
    create_llm_client,
)

UPDATE_FIELDS_TOOL = ToolSpec(
    name="update_fields",
    description="Store extracted form fields.",
    parameters={"type": "object", "properties": {"fields": {"type": "object"}}},
)


class FakeEndpoint:
    """Records the last request and returns a canned response."""

    def __init__(self, response: Any) -> None:
        self._response = response
        self.last_request: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> Any:
        self.last_request = kwargs
        return self._response


# --------------------------------------------------------------------------- #
# Chat Completions (DeepSeek / OpenAI-compatible)
# --------------------------------------------------------------------------- #


def fake_chat_sdk(message: ChatCompletionMessage, cached: int = 0) -> SimpleNamespace:
    usage = CompletionUsage.model_construct(
        prompt_tokens=20,
        completion_tokens=5,
        total_tokens=25,
        prompt_tokens_details=PromptTokensDetails.model_construct(cached_tokens=cached),
    )
    completion = ChatCompletion.model_construct(
        model="deepseek-test",
        choices=[Choice.model_construct(index=0, finish_reason="stop", message=message)],
        usage=usage,
    )
    return SimpleNamespace(chat=SimpleNamespace(completions=FakeEndpoint(completion)))


def tool_call_message(arguments: str) -> ChatCompletionMessage:
    call = ChatCompletionMessageFunctionToolCall.model_construct(
        id="call_1",
        type="function",
        function=Function.model_construct(name="update_fields", arguments=arguments),
    )
    return ChatCompletionMessage.model_construct(role="assistant", content=None, tool_calls=[call])


def test_chat_text_response_is_normalized() -> None:
    message = ChatCompletionMessage.model_construct(role="assistant", content="Merhaba!")
    sdk = fake_chat_sdk(message, cached=16)
    client = ChatCompletionsClient(
        model="deepseek-test", extra_body={"thinking": {"type": "disabled"}}, sdk_client=sdk
    )

    result = client.chat([client.user_message("Selam")], system="You are helpful.")

    assert result.text == "Merhaba!"
    assert result.tool_calls == []
    assert (result.input_tokens, result.output_tokens, result.cached_input_tokens) == (20, 5, 16)
    assert result.history_items == [{"role": "assistant", "content": "Merhaba!"}]

    request = sdk.chat.completions.last_request
    # System prompt is prepended per request, not stored in history.
    assert request["messages"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Selam"},
    ]
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "tools" not in request


def test_chat_tool_call_is_parsed_and_tools_are_converted() -> None:
    sdk = fake_chat_sdk(tool_call_message('{"fields": {"phone": "05321234567"}}'))
    client = ChatCompletionsClient(model="deepseek-test", sdk_client=sdk)

    result = client.chat([client.user_message("0532 123 45 67")], tools=[UPDATE_FIELDS_TOOL])

    call = result.tool_calls[0]
    assert (call.id, call.name) == ("call_1", "update_fields")
    assert call.arguments == {"fields": {"phone": "05321234567"}}
    assert result.history_items[0]["tool_calls"][0]["id"] == "call_1"

    sent_tool = sdk.chat.completions.last_request["tools"][0]
    assert sent_tool == {
        "type": "function",
        "function": {
            "name": "update_fields",
            "description": "Store extracted form fields.",
            "parameters": UPDATE_FIELDS_TOOL.parameters,
        },
    }


def test_chat_keeps_deepseek_reasoning_content_in_history() -> None:
    # DeepSeek returns `reasoning_content` in thinking mode and requires it
    # to be sent back when tools are used.
    message = ChatCompletionMessage.model_construct(
        role="assistant", content="Tamam.", reasoning_content="User wants an appointment."
    )
    client = ChatCompletionsClient(model="deepseek-test", sdk_client=fake_chat_sdk(message))

    result = client.chat([client.user_message("Randevu")])

    assert result.history_items[0]["reasoning_content"] == "User wants an appointment."


def test_chat_tool_result_item_format() -> None:
    client = ChatCompletionsClient(model="x", sdk_client=SimpleNamespace())
    assert client.tool_result("call_1", "ok") == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "ok",
    }


@pytest.mark.parametrize("raw", ["{not json", "[1, 2]"])
def test_invalid_tool_arguments_become_none(raw: str) -> None:
    client = ChatCompletionsClient(model="x", sdk_client=fake_chat_sdk(tool_call_message(raw)))

    call = client.chat([client.user_message("hi")]).tool_calls[0]

    assert call.arguments is None
    assert call.raw_arguments == raw


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def test_factory_builds_deepseek_client() -> None:
    client = create_llm_client(Settings(llm_provider="deepseek", deepseek_api_key="sk-test"))

    assert isinstance(client, ChatCompletionsClient)
    assert client.model == "deepseek-flash"
    assert str(client._client.base_url).rstrip("/") == DEEPSEEK_BASE_URL
    assert client._extra_body == {"thinking": {"type": "disabled"}}


@pytest.mark.parametrize(
    ("settings", "error"),
    [
        (Settings(llm_provider="deepseek"), "DEEPSEEK_API_KEY"),
        (Settings(llm_provider="foo"), "Unknown LLM_PROVIDER"),
        (
            Settings(llm_provider="deepseek", deepseek_api_key="k", deepseek_thinking="maybe"),
            "DEEPSEEK_THINKING",
        ),
    ],
)
def test_factory_fails_fast_on_bad_config(settings: Settings, error: str) -> None:
    with pytest.raises(RuntimeError, match=error):
        create_llm_client(settings)
