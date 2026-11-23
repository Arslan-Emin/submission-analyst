"""Nested Jotform submissions -> flat tables, in code (deterministic, testable).

The LLM never sees raw submission JSON. It sees tables with meaningful column
names, and for each column the original question label, so a question such as
"how satisfied are mobile users?" maps onto `satisfaction_score` and `device`.

Output:
- `submissions`: one row per submission; one column per single-valued question.
- `submission_<column>`: one row per selected option of each multi-select
  (checkbox) question, joined on `submission_id`. A list does not fit in one SQL
  cell without arrays, and arrays are where text-to-SQL models make most mistakes.

Answer shapes handled (Jotform API): plain strings, compound dicts such as
full name {"first", "last"} and lists for checkboxes. Shapes are verified against
the fixture; live API answers are checked when the REST source is added.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from analyst.sources.base import SourceData

MAIN_TABLE = "submissions"
ID_COLUMN = "submission_id"
CHILD_TABLE_PREFIX = "submission_"  # + column name of a multi-select question

# Layout elements and buttons: nothing to analyze.
NON_INPUT_TYPES = {
    "control_head",
    "control_button",
    "control_text",
    "control_divider",
    "control_pagebreak",
    "control_image",
    "control_collapse",
    "control_separator",
    "control_captcha",
}
INTEGER_TYPES = {"control_scale", "control_rating"}
NUMBER_TYPES = {"control_number", "control_spinner"}
MULTI_VALUE_TYPES = {"control_checkbox"}
# Personal data: the schema marks these columns so the prompt never shows sample values.
SENSITIVE_TYPES = {"control_fullname", "control_email", "control_phone", "control_address"}

_IDENTIFIER = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class Column:
    name: str
    sql_type: str  # VARCHAR | INTEGER | DOUBLE | TIMESTAMP
    label: str  # the original question text (or a description for metadata columns)
    jotform_type: str | None = None
    sensitive: bool = False


@dataclass(frozen=True)
class Table:
    name: str
    description: str
    columns: list[Column]
    rows: list[tuple[Any, ...]] = field(default_factory=list)


def to_snake_case(name: str) -> str:
    """`satisfactionScore` -> `satisfaction_score`; anything else becomes [a-z0-9_]."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    snake = _IDENTIFIER.sub("_", spaced.lower()).strip("_")
    if not snake or snake[0].isdigit():
        snake = f"q_{snake}"
    return snake


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        parts = [str(v).strip() for v in value.values() if str(v).strip()]
        return " ".join(parts) or None
    if isinstance(value, list):
        return "; ".join(str(v) for v in value) or None
    text = str(value).strip()
    return text or None


def _number(value: Any, cast: type[int] | type[float]) -> int | float | None:
    text = _text(value)
    if text is None:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if cast is int:
        return int(number) if number.is_integer() else None
    return number


def _options(value: Any) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    return [text for item in items if (text := _text(item))]


def _input_questions(questions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(questions.values(), key=lambda q: int(q.get("order", 0) or 0))
    return [
        q
        for q in ordered
        if str(q.get("type", "")).startswith("control_") and q["type"] not in NON_INPUT_TYPES
    ]


def _column_names(questions: list[dict[str, Any]]) -> dict[str, str]:
    """qid -> unique column name, never clashing with the fixed metadata columns."""
    taken = {ID_COLUMN, "created_at"}
    names: dict[str, str] = {}
    for question in questions:
        base = to_snake_case(str(question.get("name") or f"q{question['qid']}"))
        name, suffix = base, 2
        while name in taken:
            name, suffix = f"{base}_{suffix}", suffix + 1
        taken.add(name)
        names[str(question["qid"])] = name
    return names


def _sql_type(jotform_type: str) -> str:
    if jotform_type in INTEGER_TYPES:
        return "INTEGER"
    if jotform_type in NUMBER_TYPES:
        return "DOUBLE"
    return "VARCHAR"


def normalize(data: SourceData) -> list[Table]:
    questions = _input_questions(data.questions)
    names = _column_names(questions)
    single = [q for q in questions if q["type"] not in MULTI_VALUE_TYPES]
    multi = [q for q in questions if q["type"] in MULTI_VALUE_TYPES]

    main_columns = [
        Column(ID_COLUMN, "VARCHAR", "Unique id of the submission"),
        Column("created_at", "TIMESTAMP", "When the form was submitted"),
    ] + [
        Column(
            names[str(q["qid"])],
            _sql_type(q["type"]),
            str(q.get("text") or q.get("name")),
            jotform_type=q["type"],
            sensitive=q["type"] in SENSITIVE_TYPES,
        )
        for q in single
    ]
    main_rows: list[tuple[Any, ...]] = []
    child_rows: dict[str, list[tuple[Any, ...]]] = {str(q["qid"]): [] for q in multi}

    for submission in data.submissions:
        # Deleted submissions are not part of the data people want to analyze.
        if submission.get("status", "ACTIVE") != "ACTIVE":
            continue
        submission_id = str(submission["id"])
        answers = submission.get("answers") or {}
        row: list[Any] = [
            submission_id,
            datetime.strptime(submission["created_at"], "%Y-%m-%d %H:%M:%S"),
        ]
        for question in single:
            answer = (answers.get(str(question["qid"])) or {}).get("answer")
            if question["type"] in INTEGER_TYPES:
                row.append(_number(answer, int))
            elif question["type"] in NUMBER_TYPES:
                row.append(_number(answer, float))
            else:
                row.append(_text(answer))
        main_rows.append(tuple(row))
        for question in multi:
            qid = str(question["qid"])
            for option in _options((answers.get(qid) or {}).get("answer")):
                child_rows[qid].append((submission_id, option))

    tables = [
        Table(
            MAIN_TABLE,
            f"One row per submission of the form '{data.form_title}'.",
            main_columns,
            main_rows,
        )
    ]
    for question in multi:
        qid = str(question["qid"])
        column = names[qid]
        label = str(question.get("text") or question.get("name"))
        tables.append(
            Table(
                f"{CHILD_TABLE_PREFIX}{column}",
                f"One row per option selected in the multi-select question '{label}'. "
                f"Join with {MAIN_TABLE} on {ID_COLUMN}. Submissions that skipped the "
                "question have no rows here.",
                [
                    Column(ID_COLUMN, "VARCHAR", f"References {MAIN_TABLE}.{ID_COLUMN}"),
                    Column(column, "VARCHAR", label, jotform_type=question["type"]),
                ],
                child_rows[qid],
            )
        )
    return tables
