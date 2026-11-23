"""DuckDB layer tests: loading, and that the read-only connection holds even
without the SQL guard in front of it."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb
import pytest

from analyst.database import QueryTimeout, build_database, open_read_only, run_query
from analyst.normalize import Column, Table, normalize
from analyst.sources.fixture import FixtureSource

TABLES = [
    Table(
        "submissions",
        "test",
        [
            Column("submission_id", "VARCHAR", "id"),
            Column("created_at", "TIMESTAMP", "created"),
            Column("score", "INTEGER", "Score"),
        ],
        [("1", datetime(2026, 10, 1, 9, 0), 5), ("2", datetime(2026, 10, 2, 9, 0), None)],
    ),
    Table("submission_tags", "test", [Column("submission_id", "VARCHAR", "id")], []),
]


@pytest.fixture
def connection(tmp_path: Path):
    path = build_database(TABLES, tmp_path / "db" / "test.duckdb")
    conn = open_read_only(path)
    yield conn
    conn.close()


def test_tables_are_loaded_with_types_and_nulls(connection) -> None:
    result = run_query(connection, "SELECT * FROM submissions ORDER BY submission_id")

    assert result.columns == ["submission_id", "created_at", "score"]
    assert result.rows == [
        ("1", datetime(2026, 10, 1, 9, 0), 5),
        ("2", datetime(2026, 10, 2, 9, 0), None),
    ]
    assert run_query(connection, "SELECT count(*) FROM submission_tags").rows == [(0,)]


def test_rebuilding_replaces_the_old_file(tmp_path: Path) -> None:
    path = tmp_path / "test.duckdb"
    build_database(TABLES, path)
    build_database(TABLES[:1], path)

    with open_read_only(path) as conn:
        names = {row[0] for row in run_query(conn, "SHOW TABLES").rows}
    assert names == {"submissions"}


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM submissions",
        "DROP TABLE submissions",
        "INSERT INTO submissions VALUES ('3', NULL, 1)",
        "CREATE TABLE x (a INT)",
        "SET enable_external_access = true",
        "SET lock_configuration = false",
        "INSTALL httpfs",
        "LOAD httpfs",
    ],
)
def test_read_only_connection_rejects_writes_and_setting_changes(connection, sql: str) -> None:
    with pytest.raises(duckdb.Error):
        run_query(connection, sql)
    assert run_query(connection, "SELECT count(*) FROM submissions").rows == [(2,)]


def test_files_on_disk_cannot_be_read(connection, tmp_path: Path) -> None:
    secret = tmp_path / "secret.csv"
    secret.write_text("api_key\nsk-123\n", encoding="utf-8")

    for sql in [
        f"SELECT * FROM read_csv('{secret.as_posix()}')",
        f"SELECT * FROM '{secret.as_posix()}'",
        f"COPY submissions TO '{(tmp_path / 'out.csv').as_posix()}'",
    ]:
        # A permission error, not e.g. a parse error: the file itself is a valid CSV.
        with pytest.raises(duckdb.PermissionException):
            run_query(connection, sql)
    assert not (tmp_path / "out.csv").exists()
    with duckdb.connect() as normal:  # control: an unrestricted connection can read it
        assert normal.execute(f"SELECT * FROM read_csv('{secret.as_posix()}')").fetchall()


def test_slow_query_is_interrupted_and_connection_stays_usable(connection) -> None:
    with pytest.raises(QueryTimeout):
        run_query(connection, "SELECT count(*) FROM range(100000000000)", timeout_s=0.2)

    assert run_query(connection, "SELECT 1").rows == [(1,)]


def test_fixture_loads_end_to_end(tmp_path: Path) -> None:
    path = build_database(normalize(FixtureSource().load()), tmp_path / "fixture.duckdb")

    with open_read_only(path) as conn:
        total = run_query(conn, "SELECT count(*) FROM submissions").rows[0][0]
        joined = run_query(
            conn,
            "SELECT count(DISTINCT s.submission_id) FROM submissions s "
            "JOIN submission_features_used f USING (submission_id)",
        ).rows[0][0]
    assert total == 400
    assert 0 < joined < total  # some people skipped the checkbox question
