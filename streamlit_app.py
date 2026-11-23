"""Streamlit UI: ask a question, get an answer, a chart, the rows and the SQL behind them.

streamlit run streamlit_app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from analyst import factory
from analyst.agent import Analyst, AnalystResult
from analyst.charts import build_chart, result_frame
from analyst.config import get_settings

EXAMPLE_QUESTIONS = [
    "Which complaint topic was most common last month?",
    "What percentage of respondents with a score below 3 used mobile?",
    "How did the average satisfaction score change week by week?",
    "Show the 5 most recent negative comments.",
]

st.set_page_config(page_title="Submission Analyst", page_icon=":material/query_stats:")


@st.cache_resource(show_spinner="Loading the survey data...")
def get_analyst() -> Analyst:
    """One analyst per server process: the database is built once and opened read-only."""
    settings = get_settings()
    tracer = factory.create_tracer(settings)
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
        try:
            with st.spinner("Writing and running SQL..."):
                result = analyst.ask(question)
        except Exception as error:  # noqa: BLE001 - show provider errors instead of crashing
            st.error(f"The question could not be answered: {error.__class__.__name__}")
            return
        history.append(result)
        render_result(result)


main()
