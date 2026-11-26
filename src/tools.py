#!/usr/bin/env python3
"""The agent's tools, class-based: read-only database access (schema
introspection + guarded SQL) and chart rendering, plus the dispatcher that
turns a model tool-call into a result. The JSON schemas the model sees live
in constants.py (TOOL_SCHEMAS).

Everything database-side is deliberately read-only:
- run_sql() accepts only a single SELECT / WITH statement,
- dangerous keywords are rejected (INSERT / UPDATE / DELETE / DROP / ... even
  inside WITH ... INSERT tricks), keywords inside string literals are ignored,
- a LIMIT is appended when the model forgets one,
- results are capped at MAX_ROWS and long strings are truncated,
- sqlite queries are cut off after QUERY_TIMEOUT_SECONDS via a progress handler.

Limits and guard patterns live in constants.py. To point this at your real
application database, edit get_connection() below -- that is the only place
that needs to change (plus the two introspection helpers, marked SWAP POINT).
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Callable
from typing import Any

from constants import (
    DB_PATH,
    FORBIDDEN_RE,
    LIMIT_RE,
    MAX_CELL_CHARS,
    MAX_ROWS,
    QUERY_TIMEOUT_SECONDS,
    SELECTISH_RE,
)
from utils import make_chart as _render_chart


class SqlTools:
    """All four tools the model can call, plus the dispatcher that routes a
    tool-call to them (schemas live in constants.TOOL_SCHEMAS). Instantiated
    once by the agent; swap get_connection() and the two introspection helpers
    to target a real database."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    # -----------------------------------------------------------------------
    # SWAP POINT 1: connection
    # -----------------------------------------------------------------------
    def get_connection(self):
        """The single swap point for your real database.

        The default is the bundled sqlite demo DB. Examples for real databases:

            import psycopg2                      # pip install psycopg2-binary
            return psycopg2.connect(
                host="localhost", dbname="myapp", user="analyst_ro", password="...",
                connect_timeout=5,
                options="-c statement_timeout=8000",   # server-side query timeout
            )

            import pymysql                       # pip install pymysql
            return pymysql.connect(host="localhost", user="analyst_ro",
                                   password="...", database="myapp", read_timeout=10)

        Best practice: use a dedicated READ-ONLY database user, so that even a bug
        in guard_sql() below could never write to your data. The statement_timeout
        then becomes your second line of defence against heavy queries.
        """
        return sqlite3.connect(self.db_path, timeout=QUERY_TIMEOUT_SECONDS)

    def _quote(self, identifier: str) -> str:
        """Quote a table name so untrusted identifiers can't break out of the SQL."""
        return '"' + identifier.replace('"', '""') + '"'

    # -----------------------------------------------------------------------
    # The SQL guard: the model only ever talks to run_sql(), which goes through here
    # -----------------------------------------------------------------------
    def guard_sql(self, sql: str) -> str:
        """Validate one read-only query and make sure it carries a LIMIT."""
        body = re.sub(r"--.*?$|/\*.*?\*/", "", sql, flags=re.DOTALL | re.MULTILINE).strip()
        if not body:
            raise ValueError("empty query")
        if ";" in body.rstrip(";"):
            raise ValueError("only a single statement is allowed")
        body = body.rstrip(";").strip()
        if not SELECTISH_RE.match(body):
            raise ValueError(
                "only SELECT / WITH queries are allowed (read-only access)"
            )
        checkable = re.sub(
            r"'(?:[^']|'')*'", "''", body
        )  # ignore keywords inside literals
        hit = FORBIDDEN_RE.search(checkable)
        if hit:
            raise ValueError(
                f"keyword {hit.group(1)!r} is not allowed (read-only access)"
            )
        if not LIMIT_RE.search(checkable):
            body += f" LIMIT {MAX_ROWS}"
        return body

    # -----------------------------------------------------------------------
    # SWAP POINT 2: schema introspection (sqlite version; see docstring comments
    # for the information_schema queries a Postgres/MySQL server would use)
    # -----------------------------------------------------------------------
    def list_tables(self) -> dict[str, Any]:
        conn = self.get_connection()
        try:
            # SWAP for Postgres/MySQL:
            #   SELECT table_name FROM information_schema.tables
            #   WHERE table_schema = 'public' ORDER BY table_name
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            return {"tables": [r[0] for r in rows]}
        finally:
            conn.close()

    def describe_table(self, table: str) -> dict[str, Any]:
        conn = self.get_connection()
        try:
            if table not in self.list_tables()["tables"]:
                raise ValueError(f"unknown table {table!r}; call list_tables first")
            columns: list[dict[str, Any]] = []
            # SWAP for Postgres/MySQL: information_schema.columns / key_column_usage
            for _cid, name, ctype, notnull, _dflt, is_pk in conn.execute(
                f"PRAGMA table_info({self._quote(table)})"
            ).fetchall():
                columns.append(
                    {
                        "name": name,
                        "type": ctype or "ANY",
                        "not_null": bool(notnull),
                        "primary_key": bool(is_pk),
                    }
                )
            foreign_keys = [
                {"column": r[3], "references": f"{r[2]}.{r[4]}"}
                for r in conn.execute(
                    f"PRAGMA foreign_key_list({self._quote(table)})"
                ).fetchall()
            ]
            cur = conn.execute(f"SELECT * FROM {self._quote(table)} LIMIT 3")
            names = [d[0] for d in cur.description]
            sample = [dict(zip(names, row)) for row in cur.fetchall()]
            return {
                "table": table,
                "columns": columns,
                "foreign_keys": foreign_keys,
                "sample_rows": sample,
            }
        finally:
            conn.close()

    # -----------------------------------------------------------------------
    # The one tool the model uses for data
    # -----------------------------------------------------------------------
    def run_sql(self, sql: str) -> dict[str, Any]:
        sql = self.guard_sql(sql)
        conn = self.get_connection()
        try:
            if isinstance(conn, sqlite3.Connection):
                deadline = time.time() + QUERY_TIMEOUT_SECONDS
                conn.set_progress_handler(
                    lambda: 0 if time.time() < deadline else 1, 1000
                )
            cur = conn.execute(sql)
            names = [d[0] for d in cur.description or []]
            rows = cur.fetchmany(MAX_ROWS + 1)
        finally:
            conn.close()
        truncated = len(rows) > MAX_ROWS
        rows = rows[:MAX_ROWS]
        out = []
        for row in rows:
            out.append(
                {
                    k: (
                        v[:MAX_CELL_CHARS] + "…"
                        if isinstance(v, str) and len(v) > MAX_CELL_CHARS
                        else v
                    )
                    for k, v in zip(names, row)
                }
            )
        return {"sql": sql, "row_count": len(out), "truncated": truncated, "rows": out}

    # -----------------------------------------------------------------------
    # Chart rendering (delegates to utils.py)
    # -----------------------------------------------------------------------
    def make_chart(self, **kwargs: Any) -> dict[str, Any]:
        return _render_chart(**kwargs)

    # -----------------------------------------------------------------------
    # Dispatcher: model tool-call -> result
    # -----------------------------------------------------------------------
    def tool_funcs(self) -> dict[str, Callable[..., dict[str, Any]]]:
        return {
            "list_tables": self.list_tables,
            "describe_table": self.describe_table,
            "run_sql": self.run_sql,
            "make_chart": self.make_chart,
        }

    def run_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        func = self.tool_funcs().get(name)
        if func is None:
            return {"error": f"unknown tool {name!r}"}
        try:
            return func(**arguments)
        except TypeError as exc:
            return {"error": f"bad arguments for {name!r}: {exc}"}
        except Exception as exc:  # noqa: BLE001 - errors go back to the model, not the user
            return {"error": f"{type(exc).__name__}: {exc}"}
