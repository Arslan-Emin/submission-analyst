"""Streamlit UI: ask a question, get an answer, a chart, the rows and the SQL behind them.

streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from analyst import factory
from analyst.agent import Analyst, AnalystResult
from analyst.budget import DailyBudget
from analyst.charts import build_chart, result_frame
from analyst.config import get_settings

EXAMPLE_QUESTIONS = [
    "Which complaint topic was most common last month?",
    "What percentage of respondents with a score below 3 used mobile?",
    "How did the average satisfaction score change week by week?",
    "Show the 5 most recent negative comments.",
]

st.set_page_config(page_title="Submission Analyst", page_icon=":material/query_stats:")


def load_secrets_into_env() -> None:
    """Streamlit Community Cloud provides secrets via st.secrets; the app reads
    settings from environment variables (analyst/config.py). Copy top-level secrets
    over, without overriding variables that are already set (e.g. from .env)."""
    try:
        secrets = dict(st.secrets)
    except Exception:  # noqa: BLE001 - no secrets file locally is normal
        return
    for key, value in secrets.items():
        if isinstance(value, str | int | float | bool) and key not in os.environ:
            os.environ[key] = str(value)


load_secrets_into_env()


@st.cache_resource
def get_budget() -> DailyBudget | None:
    limit = get_settings().daily_budget_usd
    return DailyBudget(limit) if limit else None


def demo_limit_message() -> str | None:
    """Why the demo can't take another question right now, or None if it can."""
    budget = get_budget()
    if budget is not None and budget.exhausted:
        return (
            "The demo has reached its daily usage limit. Please come back tomorrow, "
            "or run it locally (see the README)."
        )
    limit = get_settings().demo_max_questions
    if limit is not None and st.session_state.get("questions_asked", 0) >= limit:
        return f"This demo allows {limit} questions per session. Reload the page to start over."
    return None


@st.cache_resource(show_spinner="Loading the survey data...")
def get_analyst() -> Analyst:
    """One analyst per server process: the database is built once and opened read-only."""
    settings = get_settings()
    tracer = factory.create_tracer(settings)
    if (budget := get_budget()) is not None:
        tracer.add_listener(budget)  # every llm_call's cost counts towards today's total
    llm = factory.create_traced_llm(settings, tracer)
    folder = Path(tempfile.mkdtemp(prefix="submission-analyst-"))
    source = factory.create_source(settings)
    return factory.create_analyst(source, folder / "analyst.duckdb", llm, tracer)


def _markdown_text(text: str) -> str:
    # Streamlit renders $...$ as LaTeX; answers often contain dollar amounts.
    return text.replace("$", "\\$")


def render_result(result: AnalystResult) -> None:
    with st.chat_message("assistant"):
        st.markdown(_markdown_text(result.answer or "(no answer)"))
        if result.assumptions:
            st.caption(f"Assumption: {_markdown_text(result.assumptions)}")

        chart = build_chart(result.chart, result.columns, result.rows)
        if chart is not None:
            st.altair_chart(chart, width="stretch")
        if result.columns:
            label = f"Result: {len(result.rows)} row{'s' if len(result.rows) != 1 else ''}"
            if result.truncated:
                label += " (capped)"
            with st.expander(label, expanded=chart is None and len(result.rows) > 1):
                st.dataframe(result_frame(result.columns, result.rows), hide_index=True)

        if result.attempts:
            with st.expander("Generated SQL"):
                for number, attempt in enumerate(result.attempts, start=1):
                    if not attempt.ok:
                        st.caption(f"Attempt {number} failed: {attempt.error}")
                    st.code(attempt.ran_sql or attempt.sql, language="sql", wrap_lines=True)

        cost = f"${result.cost_usd:.5f}" if result.cost_usd is not None else "cost unknown"
        tokens = result.input_tokens + result.output_tokens
        fixes = result.self_corrections
        st.caption(
            f"{result.llm_calls} LLM calls · {tokens:,} tokens · {cost} · "
            f"{result.duration_ms / 1000:.1f} s · "
            f"{fixes} self-correction{'s' if fixes != 1 else ''}"
        )


def main() -> None:
    st.title("Submission Analyst")
    st.caption(
        "Ask questions about Jotform form submissions in plain language. The model writes "
        "SQL; code checks it, runs it on a read-only copy and draws the chart."
    )

    try:
        analyst = get_analyst()
    except RuntimeError as error:  # e.g. DEEPSEEK_API_KEY missing
        st.error(str(error))
        st.stop()

    with st.sidebar:
        st.subheader("Data")
        st.markdown(
            f"*{analyst.form_title}*, {analyst.submission_count} submissions. "
            + (
                f"Synthetic snapshot: relative dates use **{analyst.today():%Y-%m-%d}** as today."
                if analyst.snapshot
                else "Live Jotform data: relative dates use today's date."
            )
        )
        with st.expander("Schema the model sees"):
            st.code(analyst.schema_context, language=None)

    history: list[AnalystResult] = st.session_state.setdefault("history", [])
    question = st.chat_input("Ask about the submissions...")

    if not history and not question:
        st.write("Try one of these:")
        for example in EXAMPLE_QUESTIONS:
            if st.button(example, key=f"example-{example}"):
                question = example

    for result in history:
        with st.chat_message("user"):
            st.markdown(_markdown_text(result.question))
        render_result(result)

    if question:
        with st.chat_message("user"):
            st.markdown(_markdown_text(question))
        if (limit_message := demo_limit_message()) is not None:
            st.warning(limit_message)
            return
        # Counted before the call, so a failing question still uses up the allowance.
        st.session_state.questions_asked = st.session_state.get("questions_asked", 0) + 1
        try:
            with st.spinner("Writing and running SQL..."):
                result = analyst.ask(question)
        except Exception as error:  # noqa: BLE001 - show provider errors instead of crashing
            st.error(f"The question could not be answered: {error.__class__.__name__}")
            return
        history.append(result)
        render_result(result)


main()
