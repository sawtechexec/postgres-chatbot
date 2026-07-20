"""RAG (semantic search) over the database.

A single side table, `rag_chunks`, holds one row per text chunk taken from the
source tables (emails, loxo_candidates, loxo_jobs, loxo_placements), plus its
OpenAI embedding (pgvector). `ingest.py` fills it offline; this module only
READS it, so it works fine over the app's read-only connection.

Search flow:  question -> embedding -> nearest chunks (cosine) -> LLM answer.

No pgvector client library is needed: embeddings are sent to Postgres as a
string literal with a `::vector` cast, keeping the app's dependencies minimal.
"""
from __future__ import annotations

import pandas as pd

import db

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 512
CHUNKS_TABLE = "rag_chunks"


# --------------------------------------------------------------------------- #
#  Embeddings                                                                  #
# --------------------------------------------------------------------------- #

def _client():
    from openai import OpenAI
    return OpenAI(api_key=db.get_setting("OPENAI_API_KEY"))


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. OpenAI accepts up to 2048 inputs per call."""
    # The API rejects empty strings; substitute a single space.
    cleaned = [t if t.strip() else " " for t in texts]
    resp = _client().embeddings.create(model=EMBED_MODEL, input=cleaned, dimensions=EMBED_DIM)
    return [item.embedding for item in resp.data]


def _vector_literal(vec: list[float]) -> str:
    """Format an embedding as a pgvector string literal: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{x:.6g}" for x in vec) + "]"


# --------------------------------------------------------------------------- #
#  Search (read-only)                                                          #
# --------------------------------------------------------------------------- #

def index_ready() -> bool:
    """True if the rag_chunks table exists and has at least one row."""
    try:
        df = db.run_query(
            f"SELECT 1 FROM {CHUNKS_TABLE} LIMIT 1"  # noqa: S608
        )
        return not df.empty
    except Exception:  # noqa: BLE001  (table missing, ext missing, etc.)
        return False


def index_stats() -> pd.DataFrame:
    """Chunk counts per source table (for the sidebar)."""
    return db.run_query(
        f"SELECT source_table, COUNT(*) AS chunks "  # noqa: S608
        f"FROM {CHUNKS_TABLE} GROUP BY source_table ORDER BY source_table"
    )


def search(question: str, k: int = 8, source_table: str | None = None) -> pd.DataFrame:
    """Return the k most similar chunks to the question (cosine distance)."""
    qvec = _vector_literal(embed_texts([question])[0])
    where = "WHERE source_table = %s" if source_table else ""
    sql = (
        f"SELECT source_table, source_id, chunk_index, content, "  # noqa: S608
        f"       1 - (embedding <=> %s::vector) AS similarity "
        f"FROM {CHUNKS_TABLE} {where} "
        f"ORDER BY embedding <=> %s::vector "
        f"LIMIT {int(k)}"
    )
    params: tuple = (qvec, source_table, qvec) if source_table else (qvec, qvec)
    # Direct connection (not db.run_query) so we can enable pgvector's
    # iterative scan for filtered searches: with a WHERE clause, a plain HNSW
    # scan may return only rows the filter discards, yielding empty results.
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            if source_table:
                cur.execute("SET hnsw.iterative_scan = relaxed_order")
                cur.execute("SET hnsw.max_scan_tuples = 1000000")
            cur.execute(sql, params)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns)


def _rewrite_query(question: str) -> str:
    """Turn a conversational request into a short, embedding-friendly search phrase.

    HNSW search starts in the query's semantic neighborhood; conversational
    phrasing ("find me available sres") lands among scheduling emails, while
    plain terms ("site reliability engineer") land among candidate profiles.
    """
    model = db.get_setting("OPENAI_MODEL", "gpt-4o-mini")
    resp = _client().chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": (
                "Rewrite this recruiting-database question as a short search "
                "phrase (2-6 words) describing the CONTENT to find. Expand "
                "abbreviations (sre -> site reliability engineer). Reply with "
                "ONLY the phrase.\n\nQuestion: " + question
            ),
        }],
        temperature=0,
    )
    phrase = resp.choices[0].message.content.strip().strip('"')
    return phrase or question


def answer_with_rag(question: str, k: int = 8) -> tuple[str, pd.DataFrame]:
    """Semantic search + LLM answer grounded in the retrieved chunks.

    Returns (answer_text, sources_dataframe).
    """
    # Stratified retrieval: guarantee representation for the small, high-value
    # source tables (candidates/jobs/placements), which unfiltered search
    # buries under the ~2.5M email chunks (mostly HTML boilerplate).
    query = _rewrite_query(question)
    parts = [
        search(query, k=4, source_table="loxo_candidates"),
        search(query, k=2, source_table="loxo_jobs"),
        search(query, k=2, source_table="loxo_placements"),
        search(query, k=k),
    ]
    hits = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates(subset=["source_table", "source_id", "chunk_index"])
        .sort_values("similarity", ascending=False)
        .reset_index(drop=True)
    )
    if hits.empty:
        return "No indexed content matched that question.", hits

    context = "\n\n---\n\n".join(
        f"[{r.source_table} #{r.source_id}]\n{r.content}" for r in hits.itertuples()
    )
    model = db.get_setting("OPENAI_MODEL", "gpt-4o-mini")
    prompt = (
        "Answer the user's question using ONLY the excerpts below, which come "
        "from a recruiting database (emails, candidates, jobs, placements). "
        "Cite sources inline like [emails #123]. If the excerpts don't contain "
        "the answer, say so plainly.\n\n"
        f"Excerpts:\n{context}\n\nQuestion: {question}"
    )
    resp = _client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip(), hits


def route_question(question: str) -> str:
    """Classify a question: 'sql' for counts/dates/lookups, 'search' for content."""
    model = db.get_setting("OPENAI_MODEL", "gpt-4o-mini")
    resp = _client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": (
            "Classify this question about a recruiting database. Reply with "
            "exactly one word:\n"
            "'hybrid' if it concerns consultants, placements, clients, or "
            "who is working where — EVEN IF it asks how many or for a list "
            "(these need database records PLUS email evidence, since not "
            "everything is entered in the database). Examples of 'hybrid': "
            "'how many active consultants do we have', 'list our clients', "
            "'who is placed at Citadel';\n"
            "'sql' if it asks for other counts, totals, dates, rankings, or "
            "exact record lookups;\n"
            "'search' if it asks to find people, jobs, or content by "
            "description or similarity.\n\n"
            "Question: " + question
        )}],
        temperature=0,
    )
    word = resp.choices[0].message.content.strip().lower()
    if "hybrid" in word:
        return "hybrid"
    return "sql" if "sql" in word else "search"


def summarize_rows(question: str, sql: str, df: pd.DataFrame) -> str:
    """Turn a SQL result into a short plain-English answer."""
    if df.empty:
        return "No matching records were found for that question."
    sample = df.head(50).to_csv(index=False)
    model = db.get_setting("OPENAI_MODEL", "gpt-4o-mini")
    resp = _client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": (
            "Answer the user's question concisely in plain English using this "
            "SQL result. Mention specific values/names. If the result has "
            "exactly 50 rows it may be truncated; note that if so.\n\n"
            f"Question: {question}\n\nSQL: {sql}\n\nResult (CSV):\n{sample}"
        )}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def answer_hybrid(question: str, k: int = 8) -> tuple[str, pd.DataFrame]:
    """Answer using Loxo SQL results PLUS email evidence.

    Loxo only contains what people entered; emails often reveal placements
    and clients that never made it into the database. The final answer
    distinguishes Loxo-confirmed facts from email-inferred ones.
    Returns (answer_text, email_sources_dataframe).
    """
    import queries

    # Structured baseline from Loxo.
    sql, df = "", pd.DataFrame()
    sql_note = ""
    try:
        sql, df = queries.ask_with_llm(question)
        sql_note = f"SQL used: {sql}\nResult (CSV, first 50 rows):\n" + df.head(50).to_csv(index=False)
    except Exception as exc:  # noqa: BLE001
        sql_note = f"The SQL lookup failed: {exc}"

    # Email evidence via semantic search on a rewritten query.
    query = _rewrite_query(question)
    email_hits = search(query, k=k, source_table="emails")
    email_context = "\n\n---\n\n".join(
        f"[emails #{r.source_id}]\n{r.content}" for r in email_hits.itertuples()
    ) or "(no relevant email excerpts found)"

    model = db.get_setting("OPENAI_MODEL", "gpt-4o-mini")
    prompt = (
        "You are answering a question for a recruiting firm using two sources: "
        "(1) a structured query over their Loxo database, and (2) excerpts from "
        "their email archive. Loxo is authoritative but incomplete — people do "
        "not always enter clients or placements there. Emails may reveal "
        "additional placements or clients, but are noisy (HTML fragments, "
        "scheduling boilerplate) and only suggestive.\n\n"
        "Answer the question. Present Loxo results as confirmed facts. If the "
        "emails suggest ADDITIONAL relevant placements, clients, or people not "
        "in the Loxo results, list them separately as 'Possibly per emails "
        "(not in Loxo)' with [emails #id] citations. Ignore email excerpts "
        "that are irrelevant or pure boilerplate. Do not invent anything.\n\n"
        f"--- Loxo database result ---\n{sql_note}\n\n"
        f"--- Email excerpts ---\n{email_context}\n\n"
        f"Question: {question}"
    )
    resp = _client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer = resp.choices[0].message.content.strip()
    if sql:
        answer_df = email_hits
        answer_df.attrs["sql"] = sql
        return answer, answer_df
    return answer, email_hits
