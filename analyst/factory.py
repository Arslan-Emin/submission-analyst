# Adapted from conversational-form-agent (commit 6e67866), app/factory.py:
# kept `create_tracer` and `create_traced_llm`; the form and RAG setup were dropped.
"""Shared setup for the app, the CLI and the eval runner: tracer and traced LLM."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from analyst.config import Settings, get_settings
from analyst.llm_client import LLMClient, create_llm_client
from analyst.tracing import HttpExporter, JsonlFileSink, TracedLLMClient, TraceListener, Tracer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_tracer(
    settings: Settings | None = None,
    *,
    file_name: str | None = None,
    include_content: bool | None = None,
) -> Tracer:
    """Tracer writing JSONL to TRACE_DIR (one file per UTC day unless `file_name` is given),
    and also posting the events to TRACE_HTTP_URL when that is set."""
    settings = settings or get_settings()
    if include_content is None:
        include_content = settings.trace_include_content
    if not settings.trace_enabled:
        return Tracer(include_content=include_content)
    trace_dir = Path(settings.trace_dir)
    if not trace_dir.is_absolute():
        trace_dir = PROJECT_ROOT / trace_dir
    name = file_name or f"{datetime.now(UTC):%Y-%m-%d}.jsonl"
    listeners: list[TraceListener] = [JsonlFileSink(trace_dir / name)]
    if settings.trace_http_url:
        listeners.append(HttpExporter(settings.trace_http_url, token=settings.trace_http_token))
    return Tracer(listeners, include_content=include_content)


def create_traced_llm(settings: Settings | None = None, tracer: Tracer | None = None) -> LLMClient:
    """The LLM client the app uses: provider client wrapped with tracing."""
    settings = settings or get_settings()
    return TracedLLMClient(
        create_llm_client(settings), tracer or create_tracer(settings), settings.llm_provider
    )
