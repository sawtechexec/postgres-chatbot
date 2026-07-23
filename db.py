"""Database layer: connection, schema introspection, and safe read-only queries."""
from __future__ import annotations

import os
import re
from contextlib import contextmanager

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_setting(key: str, default=None):
    """Read a setting from Streamlit secrets (cloud) or environment (.env / local).

    Streamlit Community Cloud injects secrets you set in the dashboard; locally we
    fall back to the .env file. This lets the same code run in both places.
    """
    try:
        import streamlit as st  # imported lazily so db.py works without streamlit
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:  # noqa: BLE001
        pass
    return os.getenv(key, default)


def _conn_kwargs() -> dict:
    missing = [k for k in ("PGHOST", "PGDATABASE", "PGUSER", "PGPASSWORD") if not get_setting(k)]
    if missing:
        raise RuntimeError(
            "Missing database settings: " + ", ".join(missing) +
            ". Set them in .env locally, or in the Streamlit Cloud secrets."
        )
    return dict(
        host=get_setting("PGHOST"),
        port=get_setting("PGPORT", "5432"),
        dbname=get_setting("PGDATABASE"),
        user=get_setting("PGUSER"),
        password=get_setting("PGPASSWORD"),
        sslmode=get_setting("PGSSLMODE", "require"),
        connect_timeout=10,
    )


@contextmanager
def get_connection():
    conn = psycopg2.connect(**_conn_kwargs())
    try:
        # Enforce read-only at the session level as a safety net.
        conn.set_session(readonly=True, autocommit=True)
        yield conn
    finally:
        conn.close()


def test_connection() -> tuple[bool, str]:
    """Return (ok, message) so the UI can show connection status."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                ver = cur.fetchone()[0]
        return True, ver
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


# ---- Read-only guard -------------------------------------------------------

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|"
    r"comment|copy|call|do|merge|vacuum|reindex)\b",
    re.IGNORECASE,
)


def is_safe_select(sql: str) -> tuple[bool, str]:
    """Allow a single read-only SELECT / WITH statement only."""
    stripped = sql.strip().rstrip(";")
    if not stripped:
        return False, "Empty query."
    if ";" in stripped:
        return False, "Only a single statement is allowed."
    if not re.match(r"^\s*(select|with)\b", stripped, re.IGNORECASE):
        return False, "Only SELECT / WITH queries are permitted."
    if _FORBIDDEN.search(stripped):
        return False, "Query contains a forbidden (write) keyword."
    return True, stripped


def run_query(sql: str, params: tuple | None = None, limit: int = 1000) -> pd.DataFrame:
    """Run a validated read-only query and return a DataFrame (capped by limit)."""
    ok, cleaned = is_safe_select(sql)
    if not ok:
        raise ValueError(cleaned)
    if re.search(r"\blimit\b", cleaned, re.IGNORECASE) is None:
        cleaned = f"{cleaned}\nLIMIT {int(limit)}"
    # Fetch via a psycopg2 cursor and build the DataFrame ourselves. Passing a
    # raw DBAPI connection to pandas.read_sql_query is unsupported (it warns, and
    # on some platforms the newer pandas/pyarrow stack segfaults on it).
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(cleaned, params)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns)


# ---- Schema introspection --------------------------------------------------

def list_tables() -> pd.DataFrame:
    sql = """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_type = 'BASE TABLE'
          AND table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY table_schema, table_name
    """
    return run_query(sql, limit=5000)


def list_columns(schema: str, table: str) -> pd.DataFrame:
    sql = """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
    """
    return run_query(sql, params=(schema, table), limit=5000)


def schema_summary(max_tables: int = 40) -> str:
    """A compact text summary of the schema (used for the optional LLM mode)."""
    tables = list_tables()
    lines = []
    for _, row in tables.head(max_tables).iterrows():
        cols = list_columns(row["table_schema"], row["table_name"])
        col_list = ", ".join(f"{c.column_name} {c.data_type}" for c in cols.itertuples())
        lines.append(f"{row.table_schema}.{row.table_name}({col_list})")
    return "\n".join(lines)


# ---- Feedback logging (the one intentional write) --------------------------

def log_feedback(question: str, answer: str, rating: str) -> None:
    """Insert a thumbs up/down rating into the feedback table.

    Uses its own connection because get_connection() is read-only by design.
    """
    conn = psycopg2.connect(**_conn_kwargs())
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (question, answer, rating) VALUES (%s, %s, %s)",
                (question, answer, rating),
            )
    finally:
        conn.close()
