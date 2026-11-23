# Copied from conversational-form-agent (commit 6e67866), tests/test_http_exporter.py. Unchanged.
"""HttpExporter tests: batching, retries, dropping, and never blocking the agent.

No network: httpx.MockTransport plays the eval platform's ingest endpoint.
"""

from __future__ import annotations

import json
import threading

import httpx

from analyst.config import Settings
from analyst.factory import create_tracer
from analyst.tracing import HttpExporter, Tracer

URL = "http://platform.test/v1/events"


def make_exporter(handler, **options) -> HttpExporter:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    defaults = {"flush_interval_s": 0.01, "retry_delay_s": 0.0}
    return HttpExporter(URL, client=client, **{**defaults, **options})


def test_events_are_posted_in_batches_with_the_token() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"added": 1})

    exporter = make_exporter(handler, token="s3cret", batch_size=2)
    tracer = Tracer([exporter])
    for _ in range(5):
        tracer.emit("llm_call", cost_usd=0.001)

    assert exporter.flush()
    exporter.close()

    events = [event for r in requests for event in json.loads(r.content)]
    assert len(events) == 5 and exporter.sent == 5 and exporter.dropped == 0
    assert all(len(json.loads(r.content)) <= 2 for r in requests)
    assert requests[0].headers["Authorization"] == "Bearer s3cret"
    assert {e["type"] for e in events} == {"llm_call"}


def test_server_errors_are_retried() -> None:
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(503 if len(attempts) < 3 else 200)

    exporter = make_exporter(handler, max_attempts=3)
    exporter({"event_id": "e1"})
    exporter.flush()
    exporter.close()

    assert len(attempts) == 3
    assert (exporter.sent, exporter.dropped) == (1, 0)


def test_unreachable_endpoint_drops_events_without_raising() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    exporter = make_exporter(handler, max_attempts=2)
    tracer = Tracer([exporter])
    tracer.emit("agent_turn")  # must not raise in the agent's thread

    exporter.flush()
    exporter.close()
    assert (exporter.sent, exporter.dropped) == (0, 1)


def test_client_errors_are_not_retried() -> None:
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(401)

    exporter = make_exporter(handler, max_attempts=3)
    exporter({"event_id": "e1"})
    exporter.flush()
    exporter.close()

    assert len(attempts) == 1 and exporter.dropped == 1


def test_full_queue_drops_new_events_instead_of_blocking() -> None:
    release = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        release.wait(5)  # the endpoint hangs while the agent keeps emitting
        return httpx.Response(200)

    exporter = make_exporter(handler, max_queue=2, batch_size=1)
    exporter({"event_id": "in-flight"})
    while exporter._queue.qsize():  # wait until the worker has taken it
        pass
    for i in range(5):
        exporter({"event_id": f"e{i}"})  # 2 fit in the queue, 3 are dropped

    assert exporter.dropped == 3
    release.set()
    exporter.flush()
    exporter.close()
    assert exporter.sent == 3


def test_create_tracer_adds_the_exporter_when_a_url_is_set(tmp_path) -> None:
    settings = Settings(trace_dir=str(tmp_path), trace_http_url="http://127.0.0.1:9/v1/events")
    tracer = create_tracer(settings, file_name="run.jsonl")
    exporters = [x for x in tracer._listeners if isinstance(x, HttpExporter)]
    assert len(exporters) == 1
    exporters[0].close(timeout_s=0)

    plain = create_tracer(Settings(trace_dir=str(tmp_path)), file_name="run.jsonl")
    assert not any(isinstance(x, HttpExporter) for x in plain._listeners)
