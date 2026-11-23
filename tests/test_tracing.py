# Copied from conversational-form-agent (commit 6e67866), tests/test_tracing.py.
# Changes: the form-agent turn test was dropped; this project's agent gets its own.
"""Tracing tests: event shape, privacy default, JSONL output, summaries."""

from __future__ import annotations

import json
import threading

import pytest

from analyst.config import Settings
from analyst.factory import create_tracer
from analyst.tracing import (
    InMemorySink,
    JsonlFileSink,
    TracedLLMClient,
    Tracer,
    load_events,
    percentile,
    summarize_events,
    trace_context,
)
from tests.fakes import ScriptedLLM, text_reply, tool_reply


def make_traced(responses, include_content=False):
    sink = InMemorySink()
    tracer = Tracer([sink], include_content=include_content)
    llm = TracedLLMClient(ScriptedLLM(responses), tracer, provider="fake")
    return llm, tracer, sink


def test_llm_call_event_has_tokens_latency_and_tool_names() -> None:
    llm, _, sink = make_traced([tool_reply("update_fields", {"fields": {"phone": "0532"}})])

    llm.chat([llm.user_message("hi")], tools=[])

    (event,) = sink.events
    assert event["type"] == "llm_call"
    assert event["ok"] is True
    assert (event["input_tokens"], event["output_tokens"]) == (10, 5)
    assert event["latency_ms"] == 1.0
    assert event["tool_calls"] == ["update_fields"]
    assert event["schema_version"] == 1 and event["event_id"] and event["timestamp"]


def test_content_is_not_traced_by_default() -> None:
    llm, _, sink = make_traced([tool_reply("update_fields", {"fields": {"phone": "0532"}})])

    llm.chat([llm.user_message("my phone is 0532")])

    assert "0532" not in json.dumps(sink.events)


def test_content_is_traced_when_enabled() -> None:
    llm, _, sink = make_traced([text_reply("Merhaba!")], include_content=True)

    llm.chat([llm.user_message("Selam")])

    assert sink.events[0]["output_text"] == "Merhaba!"


def test_failed_llm_call_is_traced_and_reraised() -> None:
    llm, _, sink = make_traced([RuntimeError("down")])

    with pytest.raises(RuntimeError):
        llm.chat([llm.user_message("hi")])

    assert sink.events[0]["ok"] is False
    assert sink.events[0]["error"] == "RuntimeError"


def test_broken_listener_never_breaks_the_app() -> None:
    def broken(event):
        raise ValueError("exporter down")

    sink = InMemorySink()
    tracer = Tracer([broken, sink])

    tracer.emit("custom")

    assert len(sink.events) == 1


def test_trace_context_is_attached_and_restored() -> None:
    sink = InMemorySink()
    tracer = Tracer([sink])

    with trace_context(conversation_id="c1"):
        with trace_context(turn_index=2):
            tracer.emit("inner")
        tracer.emit("outer")
    tracer.emit("none")

    inner, outer, none = sink.events
    assert (inner["conversation_id"], inner["turn_index"]) == ("c1", 2)
    assert "turn_index" not in outer and outer["conversation_id"] == "c1"
    assert "conversation_id" not in none


def test_jsonl_sink_writes_one_line_per_event_from_many_threads(tmp_path) -> None:
    path = tmp_path / "traces" / "t.jsonl"
    tracer = Tracer([JsonlFileSink(path)])

    threads = [threading.Thread(target=tracer.emit, args=("e",)) for _ in range(50)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    events = load_events(path)
    assert len(events) == 50
    assert len({e["event_id"] for e in events}) == 50


def test_create_tracer_respects_settings(tmp_path) -> None:
    disabled = create_tracer(Settings(trace_enabled=False))
    assert disabled.emit("x")["type"] == "x"  # works, writes nothing

    enabled = create_tracer(Settings(trace_dir=str(tmp_path)), file_name="run.jsonl")
    enabled.emit("x")
    assert len(load_events(tmp_path / "run.jsonl")) == 1


def test_percentile_nearest_rank() -> None:
    values = [100, 200, 300, 400, 1000]
    assert percentile(values, 50) == 300
    assert percentile(values, 95) == 1000
    assert percentile([], 50) is None


def test_summary_of_events() -> None:
    events = [
        {
            "type": "llm_call",
            "ok": True,
            "model": "m",
            "conversation_id": "a",
            "input_tokens": 100,
            "cached_input_tokens": 50,
            "output_tokens": 10,
            "cost_usd": 0.001,
            "latency_ms": 800,
        },
        {
            "type": "llm_call",
            "ok": True,
            "model": "m",
            "conversation_id": "b",
            "input_tokens": 100,
            "cached_input_tokens": 0,
            "output_tokens": 10,
            "cost_usd": None,
            "latency_ms": 1200,
        },
        {"type": "llm_call", "ok": False, "model": "m", "conversation_id": "b", "latency_ms": 5},
        {"type": "tool_call", "name": "update_fields", "ok": False, "conversation_id": "b"},
        {"type": "agent_turn", "ok": True, "conversation_id": "a", "latency_ms": 2000},
    ]

    summary = summarize_events(events)

    assert summary["conversations"] == 2
    assert (summary["llm_calls"], summary["llm_errors"]) == (3, 1)
    assert summary["input_tokens"] == 200
    assert summary["cost_usd"] == 0.001
    assert summary["calls_without_price"] == 1
    assert summary["llm_latency_ms"]["p50"] == 800
    assert summary["tools"] == {"update_fields": {"calls": 1, "failed": 1}}
