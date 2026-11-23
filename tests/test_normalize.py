"""Normalization tests: nested Jotform answers -> flat, typed tables."""

from __future__ import annotations

from datetime import datetime

import pytest

from analyst.normalize import normalize, to_snake_case
from analyst.sources.base import SourceData
from analyst.sources.fixture import FixtureSource

QUESTIONS = {
    "1": {"qid": "1", "order": "1", "type": "control_head", "name": "header", "text": "Survey"},
    "3": {"qid": "3", "order": "2", "type": "control_fullname", "name": "fullName", "text": "Name"},
    "5": {"qid": "5", "order": "3", "type": "control_scale", "name": "satisfactionScore",
          "text": "How satisfied are you?"},
    "9": {"qid": "9", "order": "4", "type": "control_checkbox", "name": "featuresUsed",
          "text": "Which features?"},
    "10": {"qid": "10", "order": "5", "type": "control_textbox", "name": "city", "text": "City"},
    "2": {"qid": "2", "order": "6", "type": "control_button", "name": "submit", "text": "Go"},
}  # fmt: skip


def submission(sid: str, answers: dict[str, object], status: str = "ACTIVE") -> dict:
    return {
        "id": sid,
        "created_at": "2026-10-05 14:30:00",
        "status": status,
        "answers": {qid: {"answer": value} for qid, value in answers.items()},
    }


def make(submissions: list[dict]) -> SourceData:
    return SourceData("f1", "Survey", QUESTIONS, submissions)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("satisfactionScore", "satisfaction_score"),
        ("NPS score", "nps_score"),
        ("city", "city"),
        ("3rd choice", "q_3rd_choice"),
        ("Şehir?", "ehir"),
        ("", "q_"),
    ],
)
def test_to_snake_case(raw: str, expected: str) -> None:
    assert to_snake_case(raw) == expected


def test_main_table_columns_follow_form_order_and_skip_layout_and_checkbox() -> None:
    main, child = normalize(make([]))

    assert main.name == "submissions"
    assert [c.name for c in main.columns] == [
        "submission_id",
        "created_at",
        "full_name",
        "satisfaction_score",
        "city",
    ]
    assert [c.sql_type for c in main.columns] == [
        "VARCHAR",
        "TIMESTAMP",
        "VARCHAR",
        "INTEGER",
        "VARCHAR",
    ]
    by_name = {c.name: c for c in main.columns}
    assert by_name["satisfaction_score"].label == "How satisfied are you?"
    assert by_name["full_name"].sensitive is True
    assert by_name["city"].sensitive is False
    assert child.name == "submission_features_used"
    assert [c.name for c in child.columns] == ["submission_id", "features_used"]


def test_answers_are_flattened_and_typed() -> None:
    main, child = normalize(
        make(
            [
                submission(
                    "1",
                    {
                        "3": {"first": "Ada", "middle": "", "last": "Kaya"},
                        "5": "4",
                        "9": ["Reports", "Payments"],
                        "10": "  Izmir ",
                    },
                )
            ]
        )
    )

    assert main.rows == [("1", datetime(2026, 10, 5, 14, 30), "Ada Kaya", 4, "Izmir")]
    assert child.rows == [("1", "Reports"), ("1", "Payments")]


def test_missing_empty_and_invalid_answers_become_null() -> None:
    main, child = normalize(make([submission("1", {"3": {"first": "", "last": ""}, "5": "n/a"})]))

    assert main.rows == [("1", datetime(2026, 10, 5, 14, 30), None, None, None)]
    assert child.rows == []


def test_deleted_submissions_are_skipped() -> None:
    main, _ = normalize(make([submission("1", {}), submission("2", {}, status="DELETED")]))

    assert [row[0] for row in main.rows] == ["1"]


def test_duplicate_question_names_get_unique_columns() -> None:
    questions = {
        "1": {"qid": "1", "order": "1", "type": "control_textbox", "name": "city", "text": "A"},
        "2": {"qid": "2", "order": "2", "type": "control_textbox", "name": "City", "text": "B"},
        "3": {
            "qid": "3",
            "order": "3",
            "type": "control_textbox",
            "name": "createdAt",
            "text": "C",
        },
    }
    (main,) = normalize(SourceData("f", "t", questions, []))

    assert [c.name for c in main.columns][2:] == ["city", "city_2", "created_at_2"]


def test_fixture_normalizes_to_two_tables() -> None:
    main, features = normalize(FixtureSource().load())

    assert len(main.rows) == 400
    names = [c.name for c in main.columns]
    assert names == [
        "submission_id",
        "created_at",
        "full_name",
        "email",
        "satisfaction_score",
        "nps_score",
        "device",
        "complaint_topic",
        "city",
        "comment",
    ]
    assert all(1 <= row[4] <= 5 for row in main.rows)
    assert features.name == "submission_features_used"
    assert len(features.rows) > len(main.rows)  # multi-select: more rows than submissions
