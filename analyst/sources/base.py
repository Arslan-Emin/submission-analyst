"""The data source contract: every source returns a form's questions and submissions
in Jotform's own API shape, so normalization and the agent never depend on where
the data came from (a local fixture, the REST API or the MCP server)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol


@dataclass(frozen=True)
class SourceData:
    form_id: str
    form_title: str
    # GET /form/{id}/questions: {qid: question}
    questions: dict[str, dict[str, Any]]
    # GET /form/{id}/submissions: each with an `answers` dict keyed by qid
    submissions: list[dict[str, Any]]
    # The moment the data is a snapshot of. "Today" for relative dates ("last month")
    # is this moment; None means live data, where today is the real date.
    as_of: datetime | None = None


class DataSource(Protocol):
    def load(self) -> SourceData: ...
