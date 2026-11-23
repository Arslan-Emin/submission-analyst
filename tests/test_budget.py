# Copied from conversational-form-agent (commit 6e67866), tests/test_budget.py (budget tests only).
from __future__ import annotations

from datetime import UTC, datetime

from analyst.budget import DailyBudget
from analyst.tracing import Tracer


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 11, 21, 10, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def test_budget_adds_up_llm_call_costs_until_the_limit() -> None:
    budget = DailyBudget(limit_usd=0.01, clock=FakeClock())
    tracer = Tracer([budget])

    tracer.emit("llm_call", cost_usd=0.004)
    tracer.emit("llm_call", cost_usd=0.004)
    assert not budget.exhausted

    tracer.emit("llm_call", cost_usd=0.004)
    assert budget.exhausted
    assert round(budget.spent_today_usd, 6) == 0.012


def test_budget_ignores_other_events_and_unknown_prices() -> None:
    budget = DailyBudget(limit_usd=0.01, clock=FakeClock())
    tracer = Tracer([budget])

    tracer.emit("agent_turn", cost_usd=1.0)  # turns repeat the cost of their llm_calls
    tracer.emit("llm_call", cost_usd=None)  # model without a price
    tracer.emit("llm_call", ok=False)  # failed call

    assert budget.spent_today_usd == 0.0


def test_budget_resets_at_midnight_utc() -> None:
    clock = FakeClock()
    budget = DailyBudget(limit_usd=0.01, clock=clock)
    budget({"type": "llm_call", "cost_usd": 0.02})
    assert budget.exhausted

    clock.now = datetime(2026, 11, 22, 0, 0, 1, tzinfo=UTC)

    assert not budget.exhausted
    assert budget.spent_today_usd == 0.0
