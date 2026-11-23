"""DuckDB storage: load the normalized tables once, then query them read-only.

Defense in depth for SQL written by an LLM. The SQL guard (analyst/sql_guard.py)
is the first layer; this module is the second, and works even if the guard has a bug:
- the database file is opened with `read_only=True`: no INSERT/UPDATE/DELETE/DDL,
- `enable_external_access=False`: no reading files or URLs (read_csv, COPY, httpfs)
  and no installing or loading extensions,
- `lock_configuration=True`: SQL cannot switch those settings back with SET,
- every query has a timeout; a runaway query is interrupted.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from analyst.normalize import Table

DEFAULT_TIMEOUT_S = 5.0

READ_ONLY_CONFIG: dict[str, Any] = {
    "enable_external_access": False,
    "autoload_known_extensions": False,
    "autoinstall_known_extensions": False,
    # Must come last in spirit: once locked, nothing can be changed at runtime.
    "lock_configuration": True,
}


class QueryTimeout(RuntimeError):
    """The query ran longer than the allowed time and was interrupted."""


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[tuple[Any, ...]]


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def build_database(tables: list[Table], path: Path | str) -> Path:
    """Write the tables into a new DuckDB file (replacing an existing one)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for stale in (path, path.with_name(path.name + ".wal")):
        stale.unlink(missing_ok=True)
    with duckdb.connect(str(path)) as connection:
        for table in tables:
            columns = ", ".join(f"{_quote(c.name)} {c.sql_type}" for c in table.columns)
            connection.execute(f"CREATE TABLE {_quote(table.name)} ({columns})")
            if table.rows:
                placeholders = ", ".join("?" for _ in table.columns)
                connection.executemany(
                    f"INSERT INTO {_quote(table.name)} VALUES ({placeholders})",
                    [list(row) for row in table.rows],
                )
    return path


def open_read_only(path: Path | str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(path), read_only=True, config=dict(READ_ONLY_CONFIG))


def run_query(
    connection: duckdb.DuckDBPyConnection, sql: str, *, timeout_s: float = DEFAULT_TIMEOUT_S
) -> QueryResult:
    """Run one query with a timeout.

    Each query gets its own cursor, so concurrent users (Streamlit sessions) never
    share one in-flight query, and the timeout interrupts only this query.
    DuckDB errors (syntax, unknown column, permission) are raised unchanged: the
    agent sends their message back to the LLM to fix the SQL.
    """
    cursor = connection.cursor()
    timer = threading.Timer(timeout_s, cursor.interrupt)
    timer.start()
    try:
        cursor.execute(sql)
        columns = [d[0] for d in cursor.description or []]
        rows = cursor.fetchall()
    except duckdb.InterruptException as error:
        raise QueryTimeout(f"Query took longer than {timeout_s:g} s and was stopped.") from error
    finally:
        timer.cancel()
        cursor.close()
    return QueryResult(columns=columns, rows=rows)
