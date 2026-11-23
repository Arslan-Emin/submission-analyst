"""Settings tests: environment variables win, empty values mean "not set"."""

from __future__ import annotations

import pytest

from analyst.config import Settings, get_settings

ENV_NAMES = [
    "LLM_PROVIDER",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_THINKING",
    "JOTFORM_API_KEY",
    "JOTFORM_FORM_ID",
    "JOTFORM_BASE_URL",
    "TRACE_ENABLED",
    "TRACE_DIR",
    "TRACE_INCLUDE_CONTENT",
    "TRACE_HTTP_URL",
    "TRACE_HTTP_TOKEN",
    "DEMO_MAX_QUESTIONS",
    "DAILY_BUDGET_USD",
]


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    # Never read a developer's real .env. (Changing the working directory would not
    # help: load_dotenv() searches upwards from the calling module's folder.)
    monkeypatch.setattr("analyst.config.load_dotenv", lambda: False)
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_defaults_without_environment(clean_env: pytest.MonkeyPatch) -> None:
    settings = get_settings()

    assert settings == Settings()
    assert settings.deepseek_model == "deepseek-flash"
    assert settings.deepseek_thinking == "disabled"
    assert settings.trace_include_content is False
    assert settings.jotform_configured is False


def test_environment_values_are_read(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("DEEPSEEK_THINKING", "ENABLED")
    clean_env.setenv("TRACE_ENABLED", "false")
    clean_env.setenv("JOTFORM_API_KEY", "k")
    clean_env.setenv("JOTFORM_FORM_ID", "123")
    clean_env.setenv("TRACE_HTTP_URL", "")  # empty means not set

    settings = get_settings()

    assert settings.deepseek_thinking == "enabled"
    assert settings.trace_enabled is False
    assert settings.jotform_configured is True
    assert settings.trace_http_url is None


def test_demo_limits_are_off_unless_configured(clean_env: pytest.MonkeyPatch) -> None:
    assert (get_settings().demo_max_questions, get_settings().daily_budget_usd) == (None, None)

    clean_env.setenv("DEMO_MAX_QUESTIONS", "10")
    clean_env.setenv("DAILY_BUDGET_USD", "0.5")

    assert (get_settings().demo_max_questions, get_settings().daily_budget_usd) == (10, 0.5)
