# Postgres Data Chatbot

A Streamlit web app that chats with your AWS RDS Postgres database. Ask questions
in **plain English** and it translates them to SQL with OpenAI, runs them
read-only, and shows the results.

## Features

- Web chat UI (Streamlit) with connection status and schema browser.
- Plain-English questions → SQL via OpenAI, answered against your live schema.
- All queries run **read-only** — the session is read-only and every statement
  (including LLM-generated SQL) is validated to be a single `SELECT`/`WITH`
  before it runs.
- Auto-charts simple two-column results.

> A library of reusable, schema-driven query helpers (preview rows, count rows,
> top values, numeric stats, most-recent rows) lives in `queries.py`. They are
> not surfaced in the UI, but you can import and use them directly.

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

3. Add your `OPENAI_API_KEY` to `.env` — it powers the plain-English chat. If
   it's blank, the app connects but the chat shows a "natural-language mode is
   off" notice.

## Run

```bash
streamlit run app.py
```

It opens at http://localhost:8501.

## Files

- `app.py` — Streamlit UI (plain-English chat).
- `db.py` — connection, schema introspection, read-only query guard.
- `queries.py` — OpenAI text-to-SQL, plus reusable schema-driven query helpers.
- `.env.example` — configuration template.

## Deployment

See `DEPLOY.md` for Streamlit Community Cloud. Two notes learned the hard way:

- Set the app's **Python version to 3.13** in the dashboard and keep the pinned
  versions in `requirements.txt` — the unpinned bleeding-edge stack (pandas 3.0
  / numpy 2.5 on Python 3.14) segfaults on Streamlit Cloud's Linux servers.
- Put secrets (DB credentials, `OPENAI_API_KEY`, `APP_PASSWORD`) in the app's
  **Settings → Secrets** box, not in `.env` (which is never uploaded).

## Safety

The database session is opened `readonly=True`, and every query (including
LLM-generated SQL) must pass `is_safe_select()`, which rejects anything that
isn't a single `SELECT`/`WITH` statement. The query helpers in `queries.py`
build identifiers with `psycopg2.sql` to avoid injection. This is
defense-in-depth, but for production also connect with a database role that only
has `SELECT` grants.
