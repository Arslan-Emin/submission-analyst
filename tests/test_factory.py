"""Data source selection."""

from __future__ import annotations

import pytest

from analyst.config import Settings
from analyst.factory import create_source
from analyst.sources.fixture import FixtureSource
from analyst.sources.jotform_rest import JotformRESTSource


def test_fixture_is_the_default_source() -> None:
    assert isinstance(create_source(Settings()), FixtureSource)


def test_jotform_source_needs_key_and_form_id() -> None:
    with pytest.raises(RuntimeError, match="JOTFORM_API_KEY and JOTFORM_FORM_ID"):
        create_source(Settings(data_source="jotform"))

    source = create_source(
        Settings(data_source="jotform", jotform_api_key="k", jotform_form_id="123")
    )
    assert isinstance(source, JotformRESTSource) and source.form_id == "123"


def test_unknown_source_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="Unknown DATA_SOURCE"):
        create_source(Settings(data_source="csv"))
