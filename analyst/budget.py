# Copied from conversational-form-agent (commit 6e67866), app/budget.py. Import path changed.
"""Daily spending cap for the public demo.

`DailyBudget` is a trace listener: it adds up the `cost_usd` of every `llm_call`
event and reports when today's total reaches the limit. The app checks it before
each message, so a public demo with a real API key cannot run up an open-ended bill.

The total lives in memory and resets at midnight UTC or when the server restarts.
That is acceptable for a demo: the API account's prepaid credit is the hard ceiling,
this is the soft one that keeps the demo from draining it in a day.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import UTC, date, datetime

from analyst.tracing import TraceEvent


class DailyBudget:
    def __init__(
        self, limit_usd: float, clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    ) -> None:
        self.limit_usd = limit_usd
        self._clock = clock
        self._day: date = clock().date()
        self._spent = 0.0
        self._lock = threading.Lock()  # events arrive from many Streamlit sessions

    def __call__(self, event: TraceEvent) -> None:
        if event.get("type") != "llm_call" or not event.get("cost_usd"):
            return
        with self._lock:
            self._roll_over()
            self._spent += float(event["cost_usd"])

    def _roll_over(self) -> None:
        today = self._clock().date()
        if today != self._day:
            self._day, self._spent = today, 0.0

    @property
    def spent_today_usd(self) -> float:
        with self._lock:
            self._roll_over()
            return self._spent

    @property
    def exhausted(self) -> bool:
        return self.spent_today_usd >= self.limit_usd
