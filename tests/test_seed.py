"""Synthetic data tests: deterministic, committed fixture up to date, nothing real."""

from __future__ import annotations

import json
from collections import Counter

from analyst.sources.fixture import DEFAULT_FIXTURE, FixtureSource
from scripts import seed


def test_same_seed_gives_same_data_and_another_seed_differs() -> None:
    assert seed.generate() == seed.generate()
    assert seed.generate(seed=1) != seed.generate()


def test_committed_fixture_matches_the_generator() -> None:
    # If this fails, run `python scripts/seed.py` and commit the new fixture.
    # Compared as parsed JSON: git may change line endings on checkout (Windows).
    committed = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    assert committed == json.loads(seed.dump(seed.generate()))


def test_fixture_source_loads_the_jotform_shapes() -> None:
    data = FixtureSource().load()

    assert data.form_title == "Customer Satisfaction Survey"
    assert len(data.submissions) == seed.SUBMISSION_COUNT
    first = data.submissions[0]
    assert set(first["answers"]) == set(data.questions)
    # Newest first, like the Jotform API.
    created = [s["created_at"] for s in data.submissions]
    assert created == sorted(created, reverse=True)


def test_data_is_synthetic() -> None:
    data = seed.generate()
    emails = [s["answers"]["4"]["answer"] for s in data["submissions"]]
    assert all(email.endswith("@example.com") for email in emails)
    assert len(set(emails)) == len(emails)


def test_prompt_injection_comments_are_planted() -> None:
    comments = [s["answers"]["11"].get("answer") for s in seed.generate()["submissions"]]
    for text in seed.INJECTION_COMMENTS:
        assert comments.count(text) == 1


def test_october_has_a_performance_spike() -> None:
    october = [
        s["answers"]["8"]["answer"]
        for s in seed.generate()["submissions"]
        if s["created_at"].startswith("2026-10")
    ]
    top_complaint = Counter(t for t in october if t != seed.NO_ISSUE).most_common(1)[0][0]
    assert top_complaint == "Performance"


def test_fixture_has_a_snapshot_date_for_relative_questions() -> None:
    data = FixtureSource().load()

    assert data.as_of == seed.AS_OF
    assert max(s["created_at"] for s in data.submissions) < data.as_of.isoformat(sep=" ")
