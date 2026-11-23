"""Send a sample of the synthetic fixture to a real Jotform form (the REST source's demo data).

Every submission counts against the account's monthly quota (100 on the free plan,
and deleting does not give it back), so nothing is sent without --send:

    python -m scripts.upload_sample --count 20                # show what would be sent
    python -m scripts.upload_sample --count 1 --send          # send one and read it back
    python -m scripts.upload_sample --count 20 --skip 1 --send

Questions are matched by their unique name (fixture `satisfactionScore` ->
the live form's question with the same name), so the live form's question ids
can differ from the fixture's. The live form keeps Jotform's own submission
dates; the fixture's dates are not sent.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from analyst.config import get_settings
from analyst.sources.fixture import FixtureSource
from analyst.sources.jotform_rest import JotformClient, JotformError

# Jotform layout elements: never answered.
SKIPPED_TYPES = {"control_head", "control_button", "control_text", "control_divider"}


def match_questions(
    fixture_questions: dict[str, dict[str, Any]], live_questions: dict[str, dict[str, Any]]
) -> dict[str, str]:
    """fixture qid -> live qid, by unique name, else by question text (case-insensitive).

    Matching by text means the form can be built in the Jotform editor without
    setting unique names. Raises if the live form lacks a question.
    """
    live_by_name = {str(q.get("name")): qid for qid, q in live_questions.items()}
    live_by_text = {
        str(q.get("text", "")).strip().casefold(): qid
        for qid, q in live_questions.items()
        if q.get("type") not in SKIPPED_TYPES
    }
    mapping, missing = {}, []
    for qid, question in fixture_questions.items():
        if question["type"] in SKIPPED_TYPES:
            continue
        name = str(question["name"])
        text = str(question["text"]).strip().casefold()
        if name in live_by_name:
            mapping[qid] = live_by_name[name]
        elif text in live_by_text:
            mapping[qid] = live_by_text[text]
        else:
            missing.append(f"{name} ({question['text']!r})")
    if missing:
        raise ValueError(f"The live form has no question named: {', '.join(missing)}")
    return mapping


def build_payload(answers: dict[str, Any], mapping: dict[str, str]) -> dict[str, str]:
    """Form-urlencoded body for POST /form/{id}/submissions.

    Compound answers use submission[qid][part] (verified for full name in the form
    agent). Multi-select answers are sent as submission[qid][0], [1], ...: the API
    reference does not document this, so the first upload is read back to check it.
    """
    payload: dict[str, str] = {}
    for fixture_qid, live_qid in mapping.items():
        answer = (answers.get(fixture_qid) or {}).get("answer")
        if answer is None:
            continue
        if isinstance(answer, dict):
            for part, value in answer.items():
                payload[f"submission[{live_qid}][{part}]"] = str(value)
        elif isinstance(answer, list):
            for index, value in enumerate(answer):
                payload[f"submission[{live_qid}][{index}]"] = str(value)
        else:
            payload[f"submission[{live_qid}]"] = str(answer)
    return payload


def pick_sample(submissions: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Evenly spread over the fixture, so every month and device shows up."""
    step = max(1, len(submissions) // count)
    return submissions[::step][:count]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--skip", type=int, default=0, help="skip the first N of the sample")
    parser.add_argument("--send", action="store_true", help="really send (uses quota)")
    args = parser.parse_args(argv)

    settings = get_settings()
    if not settings.jotform_configured:
        print("Set JOTFORM_API_KEY and JOTFORM_FORM_ID in .env first.", file=sys.stderr)
        return 1
    client = JotformClient(settings.jotform_api_key, settings.jotform_base_url)
    form_id = settings.jotform_form_id
    fixture = FixtureSource().load()
    try:
        mapping = match_questions(fixture.questions, client.get_questions(form_id))
    except (ValueError, JotformError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    sample = pick_sample(fixture.submissions, args.count)[args.skip :]
    payloads = [build_payload(s["answers"], mapping) for s in sample]
    if not args.send:
        print(f"Dry run: {len(payloads)} submissions would be sent. First payload:")
        print(json.dumps(payloads[0], ensure_ascii=False, indent=1))
        return 0

    for number, payload in enumerate(payloads, start=1):
        content = client.create_submission(form_id, payload)
        submission_id = content.get("submissionID")
        print(f"[{number}/{len(payloads)}] sent submission {submission_id}")
        if len(payloads) == 1 and submission_id:
            stored = client.get_submission(str(submission_id))
            print(json.dumps(stored.get("answers"), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
