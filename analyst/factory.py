# Adapted from conversational-form-agent (commit 6e67866), app/factory.py:
# kept `create_tracer` and `create_traced_llm`; the form and RAG setup were dropped.
"""Shared setup for the app, the CLI and the eval runner: tracer, traced LLM, analyst."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from analyst.agent import Analyst
from analyst.config import Settings, get_settings
from analyst.database import build_database, open_read_only
from analyst.llm_client import LLMClient, create_llm_client
from analyst.normalize import normalize
from analyst.sources.base import DataSource
from analyst.sources.fixture import FixtureSource
from analyst.sources.jotform_rest import JotformClient, JotformRESTSource
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


def create_analyst(
    source: DataSource,
    db_path: Path | str,
    llm: LLMClient,
    tracer: Tracer,
    *,
    clock: Callable[[], datetime] | None = None,
) -> Analyst:
    """Load the source, build the read-only database at `db_path` and wrap it in an Analyst.

    "Today" is the source's snapshot date when it has one (the fixture), so relative
    questions such as "last month" keep their meaning; otherwise the real date.
    The caller owns `analyst.connection` and closes it.
    """
    data = source.load()
    tables = normalize(data)
    connection = open_read_only(build_database(tables, db_path))
    if clock is None:
        as_of = data.as_of
        clock = (lambda: as_of) if as_of else datetime.now
    return Analyst(
        llm,
        connection,
        tables,
        tracer=tracer,
        clock=clock,
        form_title=data.form_title,
        snapshot=data.as_of is not None,
    )


def create_source(settings: Settings | None = None) -> DataSource:
    """The configured data source. Fails fast if Jotform is chosen but not configured."""
    settings = settings or get_settings()
    if settings.data_source == "fixture":
        return FixtureSource()
    if settings.data_source == "jotform":
        if not settings.jotform_configured:
            raise RuntimeError("DATA_SOURCE=jotform needs JOTFORM_API_KEY and JOTFORM_FORM_ID.")
        client = JotformClient(settings.jotform_api_key, settings.jotform_base_url)
        return JotformRESTSource(client, settings.jotform_form_id)
    raise RuntimeError(f"Unknown DATA_SOURCE {settings.data_source!r}. Use 'fixture' or 'jotform'.")
