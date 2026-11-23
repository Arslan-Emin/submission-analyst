"""Local JSON fixture source: deterministic and offline, used by the evals and the demo."""

from __future__ import annotations

import json
from pathlib import Path

from analyst.sources.base import SourceData

DEFAULT_FIXTURE = Path(__file__).resolve().parents[2] / "data" / "fixture_submissions.json"


class FixtureSource:
    def __init__(self, path: Path | str = DEFAULT_FIXTURE) -> None:
        self.path = Path(path)

    def load(self) -> SourceData:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return SourceData(
            form_id=str(payload["form"]["id"]),
            form_title=str(payload["form"]["title"]),
            questions=payload["questions"],
            submissions=payload["submissions"],
        )
