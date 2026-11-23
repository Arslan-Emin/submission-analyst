"""SQL guard tests: what an LLM might write, and what must never run."""

from __future__ import annotations

import pytest

from analyst.database import build_database, open_read_only, run_query
from analyst.normalize import normalize
from analyst.sources.fixture import FixtureSource
from analyst.sql_guard import SQLGuardError, check_sql

TABLES = ["submissions", "submission_features_used"]


def guard(sql: str, max_rows: int = 1000):
    return check_sql(sql, TABLES, max_rows=max_rows)


PASSING = [
    "SELECT device, avg(satisfaction_score) FROM submissions GROUP BY device",
    "SELECT * FROM submissions s JOIN submission_features_used f USING (submission_id)",
    "WITH low AS (SELECT * FROM submissions WHERE satisfaction_score < 3) SELECT count(*) FROM low",
    "SELECT date_trunc('week', created_at) AS w, count(*) FROM submissions GROUP BY w",
    "SELECT device FROM submissions UNION SELECT city FROM submissions",
    "FROM submissions SELECT device",
    "SELECT count(*) FILTER (WHERE device = 'Mobile') / count(*) FROM main.submissions",
    "select * from SUBMISSIONS;",
]


@pytest.mark.parametrize("sql", PASSING)
def test_ordinary_analytics_queries_pass(sql: str) -> None:
    assert guard(sql).sql


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("DROP TABLE submissions", "Only read-only SELECT"),
        ("DELETE FROM submissions", "Only read-only SELECT"),
        ("INSERT INTO submissions VALUES (1)", "Only read-only SELECT"),
        ("UPDATE submissions SET city = 'x'", "Only read-only SELECT"),
        ("CREATE TABLE x AS SELECT 1", "Only read-only SELECT"),
        ("COPY submissions TO 'out.csv'", "Only read-only SELECT"),
        ("ATTACH 'other.db'", "Only read-only SELECT"),
        ("PRAGMA database_list", "Only read-only SELECT"),
        ("SET enable_external_access = true", "Only read-only SELECT"),
        ("INSTALL httpfs", "Only read-only SELECT"),
        ("SELECT 1; DROP TABLE submissions", "exactly one"),
        ("SELECT 1; SELECT 2", "exactly one"),
        ("", "exactly one"),
        ("SELEC device FRM submissions", "Could not parse"),
        ("SELECT * FROM read_csv('secrets.csv')", "Table functions"),
        ("SELECT * FROM read_parquet('s3://bucket/x.parquet')", "Table functions"),
        ("SELECT * FROM 'secrets.csv'", "Unknown table"),
        ("SELECT * FROM range(1000000000)", "Table functions"),
        ("SELECT * FROM users", "Unknown table"),
        ("SELECT * FROM information_schema.tables", "Only tables of this survey"),
        ("SELECT * FROM other_db.main.submissions", "Only tables of this survey"),
        ("SELECT getenv('DEEPSEEK_API_KEY')", "getenv"),
        ("SELECT current_setting('threads')", "current_setting"),
        ("SELECT * FROM submissions WHERE city IN (SELECT * FROM glob('*'))", "Table functions"),
        ("SELECT read_text('C:/secret.txt')", "read_text"),
        ("SELECT duckdb_settings()", "duckdb_settings"),
    ],
)
def test_dangerous_or_invalid_sql_is_rejected(sql: str, reason: str) -> None:
    with pytest.raises(SQLGuardError, match=reason):
        guard(sql)


def test_cte_cannot_hide_a_write() -> None:
    with pytest.raises(SQLGuardError):
        guard("WITH x AS (DELETE FROM submissions RETURNING *) SELECT * FROM x")


def test_missing_limit_is_added() -> None:
    result = guard("SELECT city FROM submissions ORDER BY city", max_rows=50)

    assert result.limit_added is True
    assert result.sql == "SELECT city FROM submissions ORDER BY city LIMIT 50"


def test_small_limit_is_kept_and_large_one_is_capped() -> None:
    small = guard("SELECT city FROM submissions LIMIT 5", max_rows=50)
    large = guard("SELECT city FROM submissions LIMIT 5000", max_rows=50)

    assert (small.limit_added, small.sql.endswith("LIMIT 5")) == (False, True)
    assert (large.limit_added, large.sql.endswith("LIMIT 50")) == (True, True)


def test_limit_applies_to_the_whole_union() -> None:
    result = guard("SELECT 1 AS a FROM submissions UNION ALL SELECT 2 FROM submissions", 10)

    assert result.sql.endswith("UNION ALL SELECT 2 FROM submissions LIMIT 10")


@pytest.fixture(scope="module")
def fixture_db(tmp_path_factory):
    path = build_database(
        normalize(FixtureSource().load()), tmp_path_factory.mktemp("db") / "f.duckdb"
    )
    connection = open_read_only(path)
    yield connection
    connection.close()


@pytest.mark.parametrize("sql", PASSING)
def test_rewritten_sql_runs_and_returns_the_same_rows(fixture_db, sql: str) -> None:
    # sqlglot re-generates the SQL (e.g. date_trunc -> DATE_TRUNC); the rewrite must
    # still be valid DuckDB with the same meaning.
    original = run_query(fixture_db, sql).rows
    rewritten = run_query(fixture_db, guard(sql).sql).rows
    assert sorted(map(repr, rewritten)) == sorted(map(repr, original[:1000]))
