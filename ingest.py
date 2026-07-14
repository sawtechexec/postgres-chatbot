"""Offline RAG ingestion: embed source-table text into `rag_chunks`.

Run this from your own machine (NOT from the deployed app — the app's
connection is read-only by design). It needs the same .env as the app, and the
DB user must be allowed to CREATE TABLE / INSERT.

Usage:
    python ingest.py --init                       # create extension + table + index
    python ingest.py --table emails --limit 500   # small test run first!
    python ingest.py --table emails               # then the full table (resumable)
    python ingest.py --all                        # all four tables

Re-running is safe: rows already ingested are skipped (anti-join on the
unique key), so you can stop and resume a big table (emails is ~939k rows)
at any time.

Tip for the big emails run: you can DROP the HNSW index before the bulk load
and re-create it after — bulk inserts are much faster without it:
    DROP INDEX IF EXISTS rag_chunks_embedding_idx;
    -- ...ingest...
    CREATE INDEX rag_chunks_embedding_idx ON rag_chunks
        USING hnsw (embedding vector_cosine_ops);
"""
from __future__ import annotations

import argparse
import sys
import time

import psycopg2
from psycopg2.extras import execute_values

import db
import rag

SOURCE_TABLES = ["emails", "loxo_candidates", "loxo_jobs", "loxo_placements"]

# Optional per-table overrides. By default every text/varchar column is
# included and the primary key is auto-detected. Example:
#   "emails": {"pk": "id", "columns": ["subject", "body", "from_addr"]},
TABLE_CONFIG: dict[str, dict] = {
    "emails": {"columns": ["from_address", "to_addresses", "subject", "body"]},
}

CHUNK_CHARS = 1500          # ~350-400 tokens per chunk
CHUNK_OVERLAP = 200
MAX_CHUNKS_PER_ROW = 3
EMBED_BATCH = 96            # texts per OpenAI embeddings call
FETCH_BATCH = 1000          # rows per DB fetch


def get_write_connection():
    """A NON-read-only connection, used only by this offline script."""
    conn = psycopg2.connect(
        host=db.get_setting("PGHOST"),
        port=db.get_setting("PGPORT", "5432"),
        dbname=db.get_setting("PGDATABASE"),
        user=db.get_setting("PGUSER"),
        password=db.get_setting("PGPASSWORD"),
        sslmode=db.get_setting("PGSSLMODE", "require"),
        connect_timeout=10,
    )
    conn.autocommit = True
    return conn


# --------------------------------------------------------------------------- #
#  Schema setup                                                                #
# --------------------------------------------------------------------------- #

def init_schema() -> None:
    ddl = f"""
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS {rag.CHUNKS_TABLE} (
        id           BIGSERIAL PRIMARY KEY,
        source_table TEXT NOT NULL,
        source_id    TEXT NOT NULL,
        chunk_index  INT  NOT NULL,
        content      TEXT NOT NULL,
        embedding    VECTOR({rag.EMBED_DIM}) NOT NULL,
        UNIQUE (source_table, source_id, chunk_index)
    );

    CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx
        ON {rag.CHUNKS_TABLE} USING hnsw (embedding vector_cosine_ops);
    CREATE INDEX IF NOT EXISTS rag_chunks_source_idx
        ON {rag.CHUNKS_TABLE} (source_table, source_id);
    """
    conn = get_write_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(ddl)
    finally:
        conn.close()
    print("Schema ready: extension, table, and indexes exist.")


# --------------------------------------------------------------------------- #
#  Table introspection                                                         #
# --------------------------------------------------------------------------- #

def detect_pk(conn, table: str) -> str:
    override = TABLE_CONFIG.get(table, {}).get("pk")
    if override:
        return override
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.attname
            FROM pg_index i
            JOIN pg_attribute a
              ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = %s::regclass AND i.indisprimary
            LIMIT 1
            """,
            (table,),
        )
        row = cur.fetchone()
    if not row:
        sys.exit(f"{table}: no primary key found — set one in TABLE_CONFIG.")
    return row[0]


def detect_text_columns(conn, table: str) -> list[str]:
    override = TABLE_CONFIG.get(table, {}).get("columns")
    if override:
        return override
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s
              AND data_type IN ('text', 'character varying', 'character')
            ORDER BY ordinal_position
            """,
            (table,),
        )
        cols = [r[0] for r in cur.fetchall()]
    if not cols:
        sys.exit(f"{table}: no text columns found — set them in TABLE_CONFIG.")
    return cols


# --------------------------------------------------------------------------- #
#  Chunking                                                                    #
# --------------------------------------------------------------------------- #

def chunk(text: str) -> list[str]:
    """Split text into overlapping character chunks."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= CHUNK_CHARS:
        return [text]
    pieces, start = [], 0
    step = CHUNK_CHARS - CHUNK_OVERLAP
    while start < len(text):
        pieces.append(text[start:start + CHUNK_CHARS])
        start += step
    return pieces


# --------------------------------------------------------------------------- #
#  Ingestion                                                                   #
# --------------------------------------------------------------------------- #

def ingest_table(table: str, limit: int | None = None) -> None:
    read_conn = get_write_connection()   # plain conn; used only for SELECTs
    read_conn.autocommit = False         # named cursors need a transaction
    write_conn = get_write_connection()
    started = time.time()
    total_rows = total_chunks = 0
    pending: list[tuple[str, int, str]] = []

    try:
        pk = detect_pk(read_conn, table)
        cols = detect_text_columns(read_conn, table)
        col_sql = ", ".join(f'"{c}"' for c in cols)
        print(f"{table}: pk={pk}, {len(cols)} text columns")

        # Anti-join skips rows that already have chunks -> resumable.
        sql = (
            f'SELECT t."{pk}"::text, {col_sql} '  # noqa: S608
            f'FROM "{table}" t '
            f"WHERE NOT EXISTS ("
            f"  SELECT 1 FROM {rag.CHUNKS_TABLE} c "
            f"  WHERE c.source_table = %s AND c.source_id = t.\"{pk}\"::text"
            f') ORDER BY t."{pk}"'
        )
        if limit:
            sql += f" LIMIT {int(limit)}"

        with read_conn.cursor(name=f"ingest_{table}") as cur:  # server-side
            cur.itersize = FETCH_BATCH
            cur.execute(sql, (table,))
            for row in cur:
                source_id, *values = row
                text = "\n".join(
                    f"{c}: {v}" for c, v in zip(cols, values) if v not in (None, "")
                )
                for i, piece in enumerate(chunk(text)[:MAX_CHUNKS_PER_ROW]):
                    pending.append((source_id, i, piece))
                total_rows += 1

                if len(pending) >= EMBED_BATCH:
                    total_chunks += _flush(write_conn, table, pending)
                    pending = []
                    rate = total_rows / max(time.time() - started, 1)
                    print(f"  {total_rows} rows, {total_chunks} chunks "
                          f"({rate:.0f} rows/s)", flush=True)
            if pending:
                total_chunks += _flush(write_conn, table, pending)
    finally:
        read_conn.close()
        write_conn.close()
    print(f"{table}: done — {total_rows} rows -> {total_chunks} chunks "
          f"in {time.time() - started:.0f}s")


def _safe_embed(texts: list[str]) -> list[list[float]]:
    """Embed texts; retry on rate limits, split in half if too large."""
    delay = 2
    while True:
        try:
            return rag.embed_texts(texts)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "maximum request size" in msg and len(texts) > 1:
                mid = len(texts) // 2
                return _safe_embed(texts[:mid]) + _safe_embed(texts[mid:])
            if "insufficient_quota" in msg:
                sys.exit("FATAL: OpenAI account is out of credits. "
                         "Add credits at platform.openai.com, then re-run.")
            if "rate_limit" in msg or "Rate limit" in msg or "429" in msg:
                print(f"  rate limited; retrying in {delay}s", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise


def _flush(conn, table: str, batch: list[tuple[str, int, str]]) -> int:
    embeddings = _safe_embed([content for _, _, content in batch])
    rows = [
        (table, sid, idx, content, rag._vector_literal(vec))  # noqa: SLF001
        for (sid, idx, content), vec in zip(batch, embeddings)
    ]
    with conn.cursor() as cur:
        execute_values(
            cur,
            f"INSERT INTO {rag.CHUNKS_TABLE} "  # noqa: S608
            f"(source_table, source_id, chunk_index, content, embedding) "
            f"VALUES %s ON CONFLICT DO NOTHING",
            rows,
            template="(%s, %s, %s, %s, %s::vector)",
        )
    return len(rows)


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init", action="store_true", help="create table + index")
    p.add_argument("--table", help="ingest one table")
    p.add_argument("--all", action="store_true", help="ingest all four tables")
    p.add_argument("--limit", type=int, help="cap rows (for a test run)")
    args = p.parse_args()

    if not db.get_setting("OPENAI_API_KEY") and not args.init:
        sys.exit("OPENAI_API_KEY is not set (needed to create embeddings).")

    if args.init:
        init_schema()
    if args.table:
        if args.table not in SOURCE_TABLES:
            sys.exit(f"Unknown table {args.table!r}. Choose from {SOURCE_TABLES}.")
        ingest_table(args.table, limit=args.limit)
    elif args.all:
        for t in SOURCE_TABLES:
            ingest_table(t, limit=args.limit)
    elif not args.init:
        p.print_help()
