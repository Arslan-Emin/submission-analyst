"""Check SQL written by the LLM before it runs. The rule lives in code, not in the prompt.

A query passes only if it is:
1. exactly one statement, and that statement is a query (SELECT, WITH ... SELECT,
   UNION / INTERSECT / EXCEPT); no INSERT, DROP, COPY, ATTACH, PRAGMA, SET, ...,
2. reading only known tables (or CTEs defined in the query itself): an allowlist,
   so file paths ('x.csv'), table functions (read_csv, range) and other schemas
   (information_schema) are all rejected without having to list them,
3. free of functions that touch files, the environment or DuckDB internals,
4. capped by a LIMIT of at most `max_rows`.

When `today` is given, functions that read the system clock (CURRENT_DATE, now(),
...) are replaced by that moment as a literal, so "last month" in SQL means the
same month as "today" in the prompt (the fixture snapshot or the eval clock),
not whatever day the query happens to run.

This is the first layer. The read-only, locked-down connection in
analyst/database.py is the second; either one alone would stop a destructive query.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

DIALECT = "duckdb"
DEFAULT_MAX_ROWS = 1000

# Node types that must never appear anywhere in the tree (also inside a CTE).
_FORBIDDEN_NODE_NAMES = [
    "Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter", "TruncateTable",
    "Copy", "Command", "Pragma", "Set", "Attach", "Detach", "Install", "Use",
    "Transaction", "Commit", "Rollback", "LoadData", "Export",
]  # fmt: skip
FORBIDDEN_NODES = tuple(getattr(exp, name) for name in _FORBIDDEN_NODE_NAMES if hasattr(exp, name))

BLOCKED_FUNCTIONS = {"getenv", "current_setting", "glob", "query", "query_table", "load"}
BLOCKED_FUNCTION_PREFIXES = ("read_", "sniff_", "parquet_", "duckdb_", "pragma_", "iceberg_")


# Functions that read the system clock, by sqlglot node type or by DuckDB name.
_CLOCK_DATE_NODES = (exp.CurrentDate,)
_CLOCK_TIME_NODES = (exp.CurrentTime,)
_CLOCK_TIMESTAMP_NODES = tuple(
    getattr(exp, name)
    for name in ("CurrentTimestamp", "CurrentDatetime", "CurrentTimestampLTZ", "Localtimestamp")
    if hasattr(exp, name)
)
_CLOCK_TIMESTAMP_FUNCTIONS = {"now", "get_current_timestamp", "transaction_timestamp"}


class SQLGuardError(ValueError):
    """The SQL was rejected. The message is safe to show to the LLM and the user."""


@dataclass(frozen=True)
class GuardedSQL:
    sql: str  # what actually runs (normalized, with the LIMIT applied)
    limit_added: bool
    clock_pinned: bool = False  # a system-clock function was replaced by `today`


def _function_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    return node.sql_name().lower()


def _pin_clock(tree: exp.Expr, today: datetime) -> tuple[exp.Expr, bool]:
    replaced = False

    def literal(value: str, sql_type: str) -> exp.Expr:
        nonlocal replaced
        replaced = True
        return exp.cast(exp.Literal.string(value), sql_type)

    def replace(node: exp.Expr) -> exp.Expr:
        if isinstance(node, _CLOCK_DATE_NODES):
            return literal(today.strftime("%Y-%m-%d"), "DATE")
        if isinstance(node, _CLOCK_TIME_NODES):
            return literal(today.strftime("%H:%M:%S"), "TIME")
        if isinstance(node, _CLOCK_TIMESTAMP_NODES) or (
            isinstance(node, exp.Anonymous) and _function_name(node) in _CLOCK_TIMESTAMP_FUNCTIONS
        ):
            return literal(today.strftime("%Y-%m-%d %H:%M:%S"), "TIMESTAMP")
        return node

    return tree.transform(replace), replaced


def check_sql(
    sql: str,
    allowed_tables: Iterable[str],
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    today: datetime | None = None,
) -> GuardedSQL:
    try:
        statements = [s for s in sqlglot.parse(sql, dialect=DIALECT) if s is not None]
    except SqlglotError as error:
        raise SQLGuardError(f"Could not parse the SQL: {str(error).splitlines()[0]}") from error
    if len(statements) != 1:
        raise SQLGuardError(f"Expected exactly one SQL statement, got {len(statements)}.")
    tree = statements[0]

    if not isinstance(tree, exp.Query):
        raise SQLGuardError(f"Only read-only SELECT queries are allowed, got {tree.key.upper()}.")
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise SQLGuardError(f"{node.key.upper()} is not allowed; only SELECT queries.")

    allowed = {name.lower() for name in allowed_tables}
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            raise SQLGuardError(
                f"Table functions are not allowed ({table.this.sql(dialect=DIALECT)}). "
                f"Query only these tables: {', '.join(sorted(allowed))}."
            )
        name = table.name.lower()
        if table.catalog or (table.db and table.db.lower() != "main"):
            raise SQLGuardError(f"Only tables of this survey can be queried, not {table.sql()}.")
        if name not in allowed and name not in cte_names:
            raise SQLGuardError(
                f"Unknown table {table.name!r}. Available tables: {', '.join(sorted(allowed))}."
            )

    for function in tree.find_all(exp.Func):
        name = _function_name(function)
        if name in BLOCKED_FUNCTIONS or name.startswith(BLOCKED_FUNCTION_PREFIXES):
            raise SQLGuardError(f"The function {name}() is not allowed.")

    clock_pinned = False
    if today is not None:
        tree, clock_pinned = _pin_clock(tree, today)

    limit = tree.args.get("limit")
    current = _int_literal(limit.expression) if isinstance(limit, exp.Limit) else None
    limit_added = current is None or current > max_rows
    if limit_added:
        tree = tree.limit(max_rows, dialect=DIALECT)
    return GuardedSQL(
        sql=tree.sql(dialect=DIALECT), limit_added=limit_added, clock_pinned=clock_pinned
    )


def _int_literal(node: exp.Expr | None) -> int | None:
    if isinstance(node, exp.Literal) and not node.is_string:
        try:
            return int(str(node.this))
        except ValueError:
            return None
    return None
