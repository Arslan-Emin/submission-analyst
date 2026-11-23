# Adapted from conversational-form-agent (commit 6e67866), app/config.py:
# kept the LLM, Jotform and tracing settings; dropped OpenAI, RAG and demo-limit settings.
"""Application settings, read from environment variables (and a local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_PROVIDER = "deepseek"
DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"
DEFAULT_JOTFORM_BASE_URL = "https://api.jotform.com"
DEFAULT_TRACE_DIR = "logs/traces"


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None or value == "" else value.lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Settings:
    llm_provider: str = DEFAULT_PROVIDER
    deepseek_api_key: str | None = None
    deepseek_model: str = DEFAULT_DEEPSEEK_MODEL
    # "disabled" = faster and cheaper; "enabled" = the model reasons before answering.
    deepseek_thinking: str = "disabled"
    jotform_api_key: str | None = None
    jotform_form_id: str | None = None
    jotform_base_url: str = DEFAULT_JOTFORM_BASE_URL
    # Where submissions come from: "fixture" (synthetic, offline) or "jotform" (REST API).
    data_source: str = "fixture"
    trace_enabled: bool = True
    trace_dir: str = DEFAULT_TRACE_DIR
    # Off by default: questions, SQL results and comments can contain personal data.
    trace_include_content: bool = False
    # Optional: also send trace events to the eval platform (`evalplat serve`),
    # e.g. http://127.0.0.1:8700/v1/events. None = JSONL files only.
    trace_http_url: str | None = None
    trace_http_token: str | None = None

    @property
    def jotform_configured(self) -> bool:
        return bool(self.jotform_api_key and self.jotform_form_id)


def get_settings() -> Settings:
    """Load settings. Real environment variables take precedence over .env values."""
    load_dotenv()
    return Settings(
        llm_provider=(os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).lower(),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY") or None,
        deepseek_model=os.getenv("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL,
        deepseek_thinking=(os.getenv("DEEPSEEK_THINKING") or "disabled").lower(),
        jotform_api_key=os.getenv("JOTFORM_API_KEY") or None,
        jotform_form_id=os.getenv("JOTFORM_FORM_ID") or None,
        jotform_base_url=os.getenv("JOTFORM_BASE_URL") or DEFAULT_JOTFORM_BASE_URL,
        data_source=(os.getenv("DATA_SOURCE") or "fixture").lower(),
        trace_enabled=_env_flag("TRACE_ENABLED", True),
        trace_dir=os.getenv("TRACE_DIR") or DEFAULT_TRACE_DIR,
        trace_include_content=_env_flag("TRACE_INCLUDE_CONTENT", False),
        trace_http_url=os.getenv("TRACE_HTTP_URL") or None,
        trace_http_token=os.getenv("TRACE_HTTP_TOKEN") or None,
    )
