# Runbook — Ship a Streamlit + Postgres Chatbot to Streamlit Cloud

A reusable, end-to-end playbook for getting a Streamlit app that talks to a
Postgres database running **locally** and **deployed on Streamlit Community
Cloud**. Written from the actual steps (and mistakes) used to ship this app, so
future changes follow a known-good path.

The project has two homes that are easy to confuse:

| | Where | Config source | How code updates |
|---|---|---|---|
| **Local** | your Mac, `streamlit run app.py` | `.env` (gitignored) | you edit files directly |
| **Cloud** | `*.streamlit.app` | Streamlit **Secrets** box | push to GitHub `main` → auto-redeploy |

The single most important rule: **the cloud app never sees `.env`.** Cloud config
lives only in the dashboard's Secrets box.

---

## Phase 1 — Get it running locally

1. **Install deps:** `pip install -r requirements.txt`
2. **Fill in `.env`** from `.env.example` (RDS host, db, user, password,
   `PGSSLMODE`, and `OPENAI_API_KEY`).
3. **Test the DB connection before launching the UI** — this isolates credential
   problems from app problems:
   ```bash
   python3 -c "import db; ok,msg=db.test_connection(); print(ok, msg[:150])"
   ```
   - `password authentication failed for user "X"` → the **username or password**
     is wrong. The server is reachable; only the credentials are rejected. Try
     the actual master user (e.g. we found it was `sawyer`, not `techexec`).
   - Connection timeout / no route → **network/security-group** problem, not
     credentials.
4. **Launch:** `streamlit run app.py` → http://localhost:8501

> ⚠️ `load_dotenv()` does **not** override a variable already set in your shell.
> If a value seems ignored, check `echo $PGPASSWORD` for a shadowing env var.

---

## Phase 2 — Make a code change and deploy it

1. **Edit locally**, then verify:
   ```bash
   python3 -c "import ast; ast.parse(open('app.py').read()); print('ok')"
   ```
   Reload http://localhost:8501 (Streamlit hot-reloads on save).
2. **Start from the deployed code, not a local copy.** If your local repo has
   diverged from GitHub, fetch and base the change on `origin/main` so you don't
   silently drop deployed-only features (we nearly wiped the `check_password()`
   gate this way):
   ```bash
   git remote add origin https://github.com/<you>/<repo>.git   # if missing
   git fetch origin
   git diff origin/main -- app.py       # confirm the diff is ONLY your change
   ```
3. **Commit and push to `main`:**
   ```bash
   git commit -am "..."
   git push origin HEAD:main
   ```
   - Auth: GitHub needs a **Personal Access Token** (not your account password).
     Create one at github.com → Settings → Developer settings → Tokens (classic)
     with the **`repo`** scope. Paste it as the password when git prompts; the
     macOS keychain remembers it after the first time.
4. **Streamlit Cloud auto-redeploys** from the push within ~1 minute.

---

## Phase 3 — Configure the cloud app (Secrets)

Local `.env` values must be re-entered in the dashboard:

1. **share.streamlit.io** → your app → **⋮ → Settings → Secrets**
2. Paste TOML (this is the shape; keep existing lines, only add what's missing):
   ```toml
   APP_PASSWORD = "..."          # app sign-in gate (check_password)

   PGHOST = "..."
   PGPORT = "5432"
   PGDATABASE = "..."
   PGUSER = "..."
   PGPASSWORD = "..."
   PGSSLMODE = "require"

   OPENAI_API_KEY = "sk-..."     # enables the plain-English chat
   OPENAI_MODEL = "gpt-4o-mini"
   ```
3. **Save changes** → app reboots (~1 min) and picks them up.

> The code (`db.get_setting`) reads `st.secrets` first, then falls back to env,
> so the same code works in both places with no changes.

---

## Phase 4 — The deployment gotcha that cost the most time

**Symptom:** the deployed app shows **"Oh no. Error running app."** after a
reboot, even though it worked before and works locally.

**Root cause:** `requirements.txt` used unpinned `>=` ranges, so a fresh cloud
rebuild resolved to the newest, least-tested releases (pandas 3.0 / numpy 2.5 on
**Python 3.14**). That combo **segfaults on Streamlit Cloud's Linux servers**
(it ran fine locally on macOS with identical versions — the crash is
platform-specific).

**How to read it:** open **Manage app** (bottom-right of the deployed app) and
look at the log tail. `run-streamlit.sh: line 9: NNN Segmentation fault` = a
native crash, not a Python traceback. Note the `Using Python 3.X` line and the
`+ pandas==` / `+ numpy==` versions it installed.

**The fix (two parts, both required):**

1. **Pin `requirements.txt`** to a stable, Linux-tested stack:
   ```
   streamlit>=1.43,<2
   psycopg2-binary>=2.9.9,<2.10
   pandas>=2.2,<2.3
   numpy>=2.1,<2.2
   python-dotenv>=1.0
   openai>=1.30,<2
   ```
2. **Set the app's Python version to 3.13** in the dashboard:
   **⋮ → Settings → General → Python version → 3.13 → Save**.
   (Required because pandas 2.2 has no Python 3.14 wheels — pinning alone would
   fail to install on 3.14.)

**Order matters:** set Python 3.13 first, *then* push the pinned requirements,
so the final rebuild installs the stable stack on the right interpreter.

**General lesson:** always pin dependencies for anything deployed. Unpinned
ranges turn every rebuild into a surprise.

---

## Phase 5 — Verify end-to-end

1. Load the `*.streamlit.app` URL → sign in with `APP_PASSWORD`.
2. Sidebar shows **Connected** + table count (proves DB + schema queries work).
3. Ask a test question, e.g. *"how many rows are in the emails table?"* →
   confirm it generates SQL and returns a result (proves the OpenAI path works).
4. Open **Manage app** logs and confirm the tail after the latest
   `🔄 Updated app!` has **no warnings, no segfault**.

---

## Quick reference — commands

```bash
# test DB creds in isolation
python3 -c "import db; ok,msg=db.test_connection(); print(ok, msg[:150])"

# list tables (sanity check schema access)
python3 -c "import db; print(db.list_tables())"

# confirm a code change is the only diff vs deployed
git fetch origin && git diff origin/main -- app.py

# ship it
git commit -am "msg" && git push origin HEAD:main
```

## What lives where (this repo)

- `app.py` — Streamlit UI (plain-English chat). Has the `check_password()` gate.
- `db.py` — connection, schema introspection, read-only guard (`is_safe_select`),
  cursor-based `run_query`.
- `queries.py` — OpenAI text-to-SQL + reusable schema-driven query helpers
  (kept as a library; not surfaced in the UI).
- `requirements.txt` — **pinned** for Linux/Streamlit-Cloud stability.
- `.env` (local only, gitignored) / Streamlit **Secrets** (cloud) — config.
