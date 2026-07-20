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


DATA_DICTIONARY = """
Business semantics and JSONB guidance (IMPORTANT — the bare schema is not enough):

loxo_placements — one row per placement (a candidate placed in a job).
  Flat columns: loxo_id, job_id, person_id, job_title, person_name, owner,
  company (often NULL — do NOT rely on it), fee (often 0 — unreliable),
  placed_at (timestamp).
  The `raw` JSONB column holds the authoritative details. Key paths:
    raw->'job_type'->>'name'            -- 'Contract' or 'Permanent'
    (raw->>'start_date')::date          -- engagement start
    (raw->>'end_date')::date            -- engagement end (may be null)
    raw->'job'->'company'->>'name'      -- client company name (use this, not the flat column)
    (raw->>'pay_rate')::numeric         -- pay rate
    (raw->>'bill_rate')::numeric        -- bill rate
    (raw->>'margin')::numeric           -- margin percent
  Definitions:
    "active consultant" / "consultant on billing" = placement where
      raw->'job_type'->>'name' = 'Contract'
      AND (raw->>'start_date')::date <= CURRENT_DATE
      AND (raw->>'end_date' IS NULL OR (raw->>'end_date')::date >= CURRENT_DATE)
    "permanent placement" = raw->'job_type'->>'name' = 'Permanent'
    "placements in <year>" = filter on placed_at (or raw start_date).

loxo_candidates — one row per candidate; loxo_jobs — one row per job
  (join placements to them via person_id/job_id -> loxo_id).

emails — ~939k recruiting emails; email_attachments — extracted attachment text.

SQL rules:
  - ALWAYS wrap OR conditions in parentheses, e.g.
    AND ( raw->>'end_date' IS NULL OR (raw->>'end_date')::date >= CURRENT_DATE )
  - For the company a consultant/placement is AT, use
    raw->'job'->'company'->>'name' from loxo_placements — NEVER
    loxo_candidates.current_company (that is stale profile text).

Always prefer the JSON paths above over guessing column names. Never invent
columns that are not in the schema or this dictionary.
"""


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
        f"Schema:\n{schema}\n{DATA_DICTIONARY}\nQuestion: {question}"
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
