# Postgres Data Chatbot

A Streamlit web app that chats with your AWS RDS Postgres database. It ships with
a set of **predefined, schema-driven queries** (safe, no LLM required) and an
**optional natural-language mode** powered by OpenAI.

## Features

- Web chat UI (Streamlit) with connection status and schema browser.
- Predefined queries that adapt to *any* schema: preview rows, count rows,
  top values in a column, numeric stats, and most-recent rows.
- All queries run **read-only** — the session is read-only and every statement
  is validated to be a single `SELECT`/`WITH` before it runs.
- Optional plain-English questions → SQL via OpenAI (only if you add an API key).
- Auto-charts simple two-column results.

## Setup

1. Install dependencies (Python 3.9+):

   ```bash
   pip install -r requirements.txt
   ```

2. Configure your database. Copy the example and fill it in:

   ```bash
   cp .env.example .env
   ```

   Edit `.env` with your RDS endpoint, database, user, and password. Keep
   `PGSSLMODE=require` for RDS.

   > **Network note:** the machine running this app must be able to reach your
   > RDS instance. That usually means the RDS security group allows inbound
   > traffic on port 5432 from your IP, or you run this from inside the same VPC.

3. (Optional) To enable plain-English questions, add your `OPENAI_API_KEY` to
   `.env`. Leave it blank to run in predefined-queries-only mode.

## Run

```bash
streamlit run app.py
```

It opens at http://localhost:8501.

## Files

- `app.py` — Streamlit UI.
- `db.py` — connection, schema introspection, read-only query guard.
- `queries.py` — predefined queries + optional OpenAI text-to-SQL.
- `.env.example` — configuration template.

## Safety

The database session is opened `readonly=True`, and every query (including
LLM-generated SQL) must pass `is_safe_select()`, which rejects anything that
isn't a single `SELECT`/`WITH` statement. Predefined queries build identifiers
with `psycopg2.sql` to avoid injection. This is defense-in-depth, but for
production also connect with a database role that only has `SELECT` grants.
