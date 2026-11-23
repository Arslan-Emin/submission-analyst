"""Streamlit smoke tests with AppTest: the page renders and answers a question
end to end (scripted LLM, real fixture database)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from analyst import factory
from analyst.pricing import PRICES, ModelPrice
from analyst.tracing import TracedLLMClient, Tracer
from tests.fakes import ScriptedLLM, text_reply

APP = Path(__file__).resolve().parent.parent / "streamlit_app.py"


@pytest.fixture
def fake_llm(monkeypatch) -> ScriptedLLM:
    plan = {
        "action": "query",
        "sql": "SELECT device, count(*) AS n FROM submissions GROUP BY device ORDER BY n DESC",
        "assumptions": "All submissions are counted.",
        "message": "",
        "chart": {"type": "bar", "x": "device", "y": "n"},
    }
    fake = ScriptedLLM([text_reply(json.dumps(plan)), text_reply("Mobile leads with $0 cost.")])
    monkeypatch.setattr(factory, "create_tracer", lambda settings: Tracer())
    monkeypatch.setattr(factory, "create_traced_llm", lambda settings, tracer: fake)
    st.cache_resource.clear()  # the analyst is cached per process
    yield fake
    st.cache_resource.clear()


def test_page_renders_examples_and_schema(fake_llm) -> None:
    at = AppTest.from_file(str(APP), default_timeout=30).run()

    assert not at.exception
    assert at.title[0].value == "Submission Analyst"
    assert len(at.button) == 4
    sidebar = at.sidebar.markdown[0].value
    assert "*Customer Satisfaction Survey*, 400 submissions." in sidebar
    assert "Synthetic snapshot" in sidebar
    assert "2026-11-23" in sidebar
    assert fake_llm.requests == []  # nothing is sent to the LLM before a question


def test_question_shows_answer_chart_rows_and_sql(fake_llm) -> None:
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    at.chat_input[0].set_value("How many per device?").run()

    assert not at.exception
    assert r"Mobile leads with \$0 cost." in [m.value for m in at.markdown]
    assert any("All submissions are counted." in c.value for c in at.caption)
    assert at.dataframe[0].value.iloc[0].tolist() == ["Mobile", 198]
    assert any("GROUP BY device" in c.value for c in at.code if c.language == "sql")
    assert len(at.get("vega_lite_chart")) == 1


def test_missing_api_key_shows_an_error(monkeypatch) -> None:
    def no_key(settings, tracer):
        raise RuntimeError("DEEPSEEK_API_KEY is not set.")

    monkeypatch.setattr(factory, "create_tracer", lambda settings: Tracer())
    monkeypatch.setattr(factory, "create_traced_llm", no_key)
    st.cache_resource.clear()

    at = AppTest.from_file(str(APP), default_timeout=30).run()

    assert at.error[0].value == "DEEPSEEK_API_KEY is not set."


def test_question_limit_stops_before_calling_the_llm(fake_llm, monkeypatch) -> None:
    monkeypatch.setenv("DEMO_MAX_QUESTIONS", "1")
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    at.chat_input[0].set_value("How many per device?").run()
    calls_after_first = len(fake_llm.requests)

    at.chat_input[0].set_value("And per city?").run()

    assert calls_after_first == 2
    assert len(fake_llm.requests) == 2  # the second question never reached the LLM
    assert "1 questions per session" in at.warning[0].value


def test_exhausted_daily_budget_blocks_questions(fake_llm, monkeypatch) -> None:
    # The fake model needs a price, or its calls cost nothing.
    monkeypatch.setitem(PRICES, "fake", ModelPrice(input=1.0, cached_input=1.0, output=1.0))
    monkeypatch.setenv("DAILY_BUDGET_USD", "0.00001")  # less than one question
    # As in the app, the LLM is wrapped with tracing: its llm_call events feed the budget.
    monkeypatch.setattr(
        factory,
        "create_traced_llm",
        lambda settings, tracer: TracedLLMClient(fake_llm, tracer, "fake"),
    )
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    at.chat_input[0].set_value("How many per device?").run()

    at.chat_input[0].set_value("And per city?").run()

    assert len(fake_llm.requests) == 2  # only the first question's two calls
    assert "daily usage limit" in at.warning[0].value
