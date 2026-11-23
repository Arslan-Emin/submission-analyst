# Copied from conversational-form-agent (commit 6e67866), app/tracing.py.
# Changes: imports point to the analyst package; TracedLLMClient forwards `response_format` and
# `max_tokens` and adds a `json_mode` field to llm_call events. Event schema otherwise unchanged.
"""Tracing: structured events for every LLM call, tool call and agent turn.

This is the hook the companion eval & observability platform (project 2) plugs
into. Every event is a flat JSON-serializable dict with a `type`, an `event_id`,
a UTC `timestamp` and the ids of the conversation and turn it belongs to.

    tracer = Tracer(listeners=[JsonlFileSink(path)])
    tracer.add_listener(HttpExporter(url))      # also send them to the eval platform

Event types:
- `llm_call`   one request to the LLM provider: model, tokens, cost, latency, tool names
- `tool_call`  one tool execution: name, ok, duration
- `agent_turn` one user message handled end to end: LLM calls, tokens, cost, latency

Privacy: message text, tool arguments and tool results can contain personal data,
so they are only included when `include_content=True` (off by default; the evals
turn it on because they only use fake data).

Run `python -m analyst.tracing summary logs/traces/<file>.jsonl` for cost/latency stats.
"""

from __future__ import annotations

import atexit
import contextvars
import json
import logging
import math
import queue
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from analyst.llm_client import HistoryItem, LLMClient, LLMResponse, ToolSpec
from analyst.pricing import estimate_cost_usd

SCHEMA_VERSION = 1
logger = logging.getLogger(__name__)

TraceEvent = dict[str, Any]
TraceListener = Callable[[TraceEvent], None]

# Fields (conversation_id, turn_index, ...) attached to every event emitted while
# they are set. contextvars keep them separate per thread / per Streamlit session.
_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "trace_context", default=None
)


@contextmanager
def trace_context(**fields: Any) -> Iterator[None]:
    token = _context.set({**(_context.get() or {}), **fields})
    try:
        yield
    finally:
        _context.reset(token)


class Tracer:
    def __init__(
        self, listeners: list[TraceListener] | None = None, *, include_content: bool = False
    ) -> None:
        self._listeners = list(listeners or [])
        self.include_content = include_content

    def add_listener(self, listener: TraceListener) -> None:
        self._listeners.append(listener)

    def emit(self, event_type: str, **fields: Any) -> TraceEvent:
        event = {
            "schema_version": SCHEMA_VERSION,
            "event_id": uuid.uuid4().hex,
            "type": event_type,
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
            **(_context.get() or {}),
            **fields,
        }
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:  # noqa: BLE001 - observability must never break the app
                logger.exception("Trace listener failed")
        return event


NULL_TRACER = Tracer()


class JsonlFileSink:
    """Appends one JSON object per line. Thread-safe; creates the folder if needed."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def __call__(self, event: TraceEvent) -> None:
        line = json.dumps(event, ensure_ascii=False, default=str)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")


class InMemorySink:
    """Collects events in a list (tests, or a UI that wants to show them)."""

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def __call__(self, event: TraceEvent) -> None:
        self.events.append(event)


class HttpExporter:
    """Sends events to an HTTP endpoint (the eval platform's `POST /v1/events`).

    Observability must never slow down or break the agent, so the listener only
    puts the event on a bounded queue; a background thread posts batches. If the
    endpoint is down, a batch is retried a few times and then dropped, and when the
    queue is full new events are dropped: both are counted in `dropped`. The JSONL
    file stays the complete record. Re-sending is safe: the platform ignores event
    ids it already has.
    """

    def __init__(
        self,
        url: str,
        *,
        token: str | None = None,
        batch_size: int = 100,
        flush_interval_s: float = 1.0,
        max_queue: int = 10_000,
        max_attempts: int = 3,
        retry_delay_s: float = 0.5,
        client: httpx.Client | None = None,
    ) -> None:
        self.url = url
        self.batch_size = batch_size
        self.flush_interval_s = flush_interval_s
        self.max_attempts = max_attempts
        self.retry_delay_s = retry_delay_s
        self.sent = 0
        self.dropped = 0
        self._headers = {"Content-Type": "application/json"}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"
        self._client = client or httpx.Client(timeout=5.0)
        self._queue: queue.Queue[TraceEvent] = queue.Queue(maxsize=max_queue)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="trace-http-exporter", daemon=True)
        self._thread.start()
        # Send what is still queued when the process exits (CLI, eval runs).
        atexit.register(self.close)

    def __call__(self, event: TraceEvent) -> None:
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self._count_dropped(1)

    def flush(self, timeout_s: float = 5.0) -> bool:
        """Wait until every queued event was sent or dropped; False on timeout."""
        deadline = time.monotonic() + timeout_s
        while self._queue.unfinished_tasks:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.01)
        return True

    def close(self, timeout_s: float = 5.0) -> None:
        if self._stop.is_set():
            return
        self.flush(timeout_s)
        self._stop.set()
        self._thread.join(timeout_s)
        self._client.close()

    # -- background thread ---------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            batch = self._next_batch()
            if not batch:
                continue
            try:
                self._send(batch)
            finally:
                for _ in batch:
                    self._queue.task_done()

    def _next_batch(self) -> list[TraceEvent]:
        try:
            batch = [self._queue.get(timeout=self.flush_interval_s)]
        except queue.Empty:
            return []
        while len(batch) < self.batch_size:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _send(self, batch: list[TraceEvent]) -> None:
        body = json.dumps(batch, ensure_ascii=False, default=str).encode("utf-8")
        problem = ""
        for attempt in range(self.max_attempts):
            try:
                response = self._client.post(self.url, content=body, headers=self._headers)
            except httpx.HTTPError as error:
                problem = error.__class__.__name__
            else:
                if response.is_success:
                    with self._lock:
                        self.sent += len(batch)
                    return
                problem = f"HTTP {response.status_code}"
                if response.status_code < 500 and response.status_code != 429:
                    break  # a client error (bad token, bad payload) will not fix itself
            if attempt + 1 < self.max_attempts:
                time.sleep(self.retry_delay_s * 2**attempt)
        logger.warning("Dropping %d trace events for %s: %s", len(batch), self.url, problem)
        self._count_dropped(len(batch))

    def _count_dropped(self, count: int) -> None:
        with self._lock:
            self.dropped += count


# --------------------------------------------------------------------------- #
# The LLM wrapper: every call made by the app goes through here.
# --------------------------------------------------------------------------- #


class TracedLLMClient:
    """Wraps any `LLMClient` and emits an `llm_call` event per request.

    A decorator rather than code inside each provider class: a new provider gets
    tracing, cost and latency tracking for free.
    """

    def __init__(self, inner: LLMClient, tracer: Tracer, provider: str) -> None:
        self._inner = inner
        self._tracer = tracer
        self.provider = provider

    @property
    def model(self) -> str:
        return self._inner.model

    def user_message(self, text: str) -> HistoryItem:
        return self._inner.user_message(text)

    def assistant_message(self, text: str) -> HistoryItem:
        return self._inner.assistant_message(text)

    def tool_result(self, call_id: str, output: str) -> HistoryItem:
        return self._inner.tool_result(call_id, output)

    def chat(
        self,
        history: list[HistoryItem],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        started = time.perf_counter()
        try:
            response = self._inner.chat(
                history,
                system=system,
                tools=tools,
                response_format=response_format,
                max_tokens=max_tokens,
            )
        except Exception as error:
            self._tracer.emit(
                "llm_call",
                provider=self.provider,
                model=self.model,
                ok=False,
                error=error.__class__.__name__,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            raise

        fields: dict[str, Any] = {
            "provider": self.provider,
            "model": response.model,
            "ok": True,
            "input_tokens": response.input_tokens,
            "cached_input_tokens": response.cached_input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": estimate_cost_usd(
                response.model,
                response.input_tokens,
                response.cached_input_tokens,
                response.output_tokens,
            ),
            "latency_ms": round(response.latency_ms, 1),
            "history_items": len(history),
            "tools_offered": [tool.name for tool in tools or []],
            "tool_calls": [call.name for call in response.tool_calls],
            "json_mode": bool(response_format),
        }
        if self._tracer.include_content:
            fields["output_text"] = response.text
            fields["tool_call_arguments"] = [call.arguments for call in response.tool_calls]
        self._tracer.emit("llm_call", **fields)
        return response


# --------------------------------------------------------------------------- #
# Reading traces back: cost and latency summary
# --------------------------------------------------------------------------- #


def load_events(path: Path | str) -> list[TraceEvent]:
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile; enough for latency reporting without numpy."""
    if not values:
        return None
    ordered = sorted(values)
    # ceil, not round: Python's round() is banker's rounding (round(2.5) == 2).
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def summarize_events(events: list[TraceEvent]) -> dict[str, Any]:
    calls = [e for e in events if e["type"] == "llm_call"]
    turns = [e for e in events if e["type"] == "agent_turn"]
    tools = [e for e in events if e["type"] == "tool_call"]
    ok_calls = [e for e in calls if e.get("ok")]
    call_latency = [e["latency_ms"] for e in ok_calls]
    turn_latency = [e["latency_ms"] for e in turns]
    known_costs = [e["cost_usd"] for e in ok_calls if e.get("cost_usd") is not None]

    tool_stats: dict[str, dict[str, int]] = {}
    for event in tools:
        stats = tool_stats.setdefault(event["name"], {"calls": 0, "failed": 0})
        stats["calls"] += 1
        stats["failed"] += not event.get("ok")

    return {
        "conversations": len({e.get("conversation_id") for e in events} - {None}),
        "llm_calls": len(calls),
        "llm_errors": len(calls) - len(ok_calls),
        "models": sorted({e["model"] for e in calls}),
        "input_tokens": sum(e.get("input_tokens", 0) for e in ok_calls),
        "cached_input_tokens": sum(e.get("cached_input_tokens", 0) for e in ok_calls),
        "output_tokens": sum(e.get("output_tokens", 0) for e in ok_calls),
        "cost_usd": round(sum(known_costs), 6),
        "calls_without_price": len(ok_calls) - len(known_costs),
        "llm_latency_ms": {
            "p50": percentile(call_latency, 50),
            "p95": percentile(call_latency, 95),
            "max": max(call_latency, default=None),
        },
        "turns": len(turns),
        "turn_latency_ms": {
            "p50": percentile(turn_latency, 50),
            "p95": percentile(turn_latency, 95),
        },
        "tools": tool_stats,
    }


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 2 or args[0] != "summary":
        print("Usage: python -m analyst.tracing summary <trace.jsonl>", file=sys.stderr)
        return 2
    print(json.dumps(summarize_events(load_events(args[1])), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
