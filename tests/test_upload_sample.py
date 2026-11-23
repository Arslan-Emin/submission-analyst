"""Upload script tests: question matching and payloads, no network."""

from __future__ import annotations

import pytest

from analyst.sources.fixture import FixtureSource
from scripts.upload_sample import build_payload, match_questions, pick_sample

LIVE = {
    "31": {"name": "fullName", "type": "control_fullname"},
    "32": {"name": "email", "type": "control_email"},
    "33": {"name": "satisfactionScore", "type": "control_scale"},
    "34": {"name": "npsScore", "type": "control_scale"},
    "35": {"name": "device", "type": "control_radio"},
    "36": {"name": "complaintTopic", "type": "control_dropdown"},
    "37": {"name": "featuresUsed", "type": "control_checkbox"},
    "38": {"name": "city", "type": "control_textbox"},
    "39": {"name": "comment", "type": "control_textarea"},
    "40": {"name": "submit2", "type": "control_button"},
}


def test_questions_are_matched_by_unique_name() -> None:
    mapping = match_questions(FixtureSource().load().questions, LIVE)

    assert mapping["5"] == "33" and mapping["9"] == "37"
    assert "1" not in mapping and "2" not in mapping  # header and button


def test_missing_live_question_is_an_error() -> None:
    live = {k: v for k, v in LIVE.items() if v["name"] != "npsScore"}
    with pytest.raises(ValueError, match="npsScore"):
        match_questions(FixtureSource().load().questions, live)


def test_payload_splits_compound_and_multi_select_answers() -> None:
    mapping = {"3": "31", "5": "33", "9": "37", "6": "34"}
    answers = {
        "3": {"answer": {"first": "Ada", "last": "Kaya"}},
        "5": {"answer": "4"},
        "9": {"answer": ["Reports", "Payments"]},
        "6": {},  # skipped question
    }

    assert build_payload(answers, mapping) == {
        "submission[31][first]": "Ada",
        "submission[31][last]": "Kaya",
        "submission[33]": "4",
        "submission[37][0]": "Reports",
        "submission[37][1]": "Payments",
    }


def test_sample_is_spread_over_the_fixture() -> None:
    submissions = FixtureSource().load().submissions
    sample = pick_sample(submissions, 20)

    assert len(sample) == 20
    months = {s["created_at"][:7] for s in sample}
    assert months == {"2026-08", "2026-09", "2026-10", "2026-11"}


def test_questions_without_unique_names_are_matched_by_text() -> None:
    fixture = FixtureSource().load().questions
    # Names as the Jotform editor generates them; texts typed by hand, other case.
    live = {
        str(50 + i): {"name": f"typeA{i}", "type": q["type"], "text": q["text"].upper()}
        for i, q in enumerate(fixture.values())
    }

    mapping = match_questions(fixture, live)

    assert mapping["5"] == str(50 + list(fixture).index("5"))
