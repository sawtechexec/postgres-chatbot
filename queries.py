"""Predefined (canned) queries plus an optional OpenAI natural-language mode.

Predefined queries are schema-driven: they build safe SQL from a table/column
the user picks in the UI, so they work against *any* Postgres schema without
you hand-writing SQL for your specific tables.
"""
from __future__ import annotations

import os

import pandas as pd
from psycopg2 import sql as _sql

import db


# --------------------------------------------------------------------------- #
#  Predefined queries                                                         #
# --------------------------------------------------------------------------- #
# Each entry: label -> function(**kwargs) -> pandas.DataFrame
# The UI collects the kwargs (table, column, limit) and calls the function.

def _ident_sql(query: _sql.Composed) -> str:
    """Render a psycopg2 Composed object to a plain SQL string for db.run_query."""
    with db.get_connection() as conn:
        return query.as_string(conn)


def preview_table(schema: str, table: str, limit: int = 50) -> pd.DataFrame:
    q = _sql.SQL("SELECT * FROM {}.{}").format(
        _sql.Identifier(schema), _sql.Identifier(table)
    )
    return db.run_query(_ident_sql(q), limit=limit)


def row_count(schema: str, table: str, **_) -> pd.DataFrame:
    q = _sql.SQL("SELECT count(*) AS row_count FROM {}.{}").format(
        _sql.Identifier(schema), _sql.Identifier(table)
    )
    return db.run_query(_ident_sql(q))


def column_value_counts(schema: str, table: str, column: str, limit: int = 20) -> pd.DataFrame:
    q = _sql.SQL(
        "SELECT {col} AS value, count(*) AS count "
        "FROM {sch}.{tbl} GROUP BY {col} ORDER BY count DESC"
    ).format(
        col=_sql.Identifier(column),
        sch=_sql.Identifier(schema),
        tbl=_sql.Identifier(table),
    )
    return db.run_query(_ident_sql(q), limit=limit)


def column_stats(schema: str, table: str, column: str, **_) -> pd.DataFrame:
    q = _sql.SQL(
        "SELECT count({col}) AS non_null, min({col}) AS min, "
        "max({col}) AS max, avg({col}::double precision) AS avg "
        "FROM {sch}.{tbl}"
    ).format(
        col=_sql.Identifier(column),
        sch=_sql.Identifier(schema),
        tbl=_sql.Identifier(table),
    )
    return db.run_query(_ident_sql(q))


def recent_rows(schema: str, table: str, column: str, limit: int = 50) -> pd.DataFrame:
    """Most-recent rows ordered by a date/timestamp column."""
    q = _sql.SQL("SELECT * FROM {sch}.{tbl} ORDER BY {col} DESC").format(
        sch=_sql.Identifier(schema),
        tbl=_sql.Identifier(table),
        col=_sql.Identifier(column),
    )
    return db.run_query(_ident_sql(q), limit=limit)


# Registry consumed by the UI. `needs` tells the UI which inputs to collect.
PREDEFINED = {
    "Preview rows (SELECT *)": {"fn": preview_table, "needs": ["table"]},
    "Count rows": {"fn": row_count, "needs": ["table"]},
    "Top values in a column": {"fn": column_value_counts, "needs": ["table", "column"]},
    "Numeric stats for a column": {"fn": column_stats, "needs": ["table", "column"]},
    "Most recent rows (by date column)": {"fn": recent_rows, "needs": ["table", "column"]},
}


# --------------------------------------------------------------------------- #
#  Optional OpenAI natural-language -> SQL                                     #
# --------------------------------------------------------------------------- #

def openai_available() -> bool:
    return bool(db.get_setting("OPENAI_API_KEY"))


def ask_with_llm(question: str) -> tuple[str, pd.DataFrame]:
    """Translate a natural-language question to SQL, run it, return (sql, df).

    Only runs if OPENAI_API_KEY is set. The generated SQL is passed through the
    same read-only guard as everything else before it touches the database.
    """
    if not openai_available():
        raise RuntimeError("OPENAI_API_KEY is not set; natural-language mode is disabled.")

    from openai import OpenAI

    client = OpenAI(api_key=db.get_setting("OPENAI_API_KEY"))
    model = db.get_setting("OPENAI_MODEL", "gpt-4o-mini")
    schema = db.schema_summary()

    prompt = (
        "You are a PostgreSQL expert. Given the schema below, write ONE read-only "
        "SQL SELECT statement that answers the user's question. Return ONLY the SQL, "
        "no explanation, no markdown fences. Never write INSERT/UPDATE/DELETE/DDL.\n\n"
        f"Schema:\n{schema}\n\nQuestion: {question}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    generated = resp.choices[0].message.content.strip()
    generated = generated.replace("```sql", "").replace("```", "").strip()

    ok, cleaned = db.is_safe_select(generated)
    if not ok:
        raise ValueError(f"Model produced an unsafe query ({cleaned}): {generated}")

    return generated, db.run_query(generated)
