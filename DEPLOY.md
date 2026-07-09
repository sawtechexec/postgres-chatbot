# Deploying to Streamlit Community Cloud

This guide puts your chatbot online with a public URL and a password login.

## Before you start

- A **GitHub account** (free) — Streamlit deploys from a GitHub repo.
- A **Streamlit Community Cloud account** (free) — sign up at
  https://share.streamlit.io with your GitHub account.

## Security: two separate locks

1. **App login** — the `APP_PASSWORD` secret gates the whole UI. Anyone without
   it just sees a password box.
2. **Database firewall** — Streamlit Cloud connects to your Postgres from *their*
   servers, and those IPs are not fixed/published. So to let the cloud app reach
   your database you have to allow inbound `5432` from a wide range. Do this as
   safely as possible:
   - Use a **strong, unique** `PGPASSWORD`.
   - Keep `PGSSLMODE=require` so traffic is encrypted.
   - Ideally create a **read-only database user** for the app (only `SELECT`
     grants), so even a leak can't change data.
   - Consider a dedicated database rather than your most sensitive one.

   If you are not comfortable opening `5432` broadly, Streamlit Community Cloud
   may not be the right host — running on your own EC2 box (where you can lock
   `5432` to `localhost`) is safer. Ask and we can switch approaches.

## Step 1 — Put the code on GitHub

Upload these files to a new GitHub repo (via the web "Add file -> Upload files"
button, or git): `app.py`, `db.py`, `queries.py`, `requirements.txt`,
`README.md`. Never upload `.env` (it holds your real password).

## Step 2 — Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io and click **Create app** / **Deploy**.
2. Pick your `postgres-chatbot` repo, branch `main`, main file `app.py`.
3. Before (or right after) deploying, open **Advanced settings -> Secrets**
   (or the app's **Settings -> Secrets** later) and paste the contents of
   `secrets.toml.example`, filled in with your real values:

   ```toml
   APP_PASSWORD = "your-strong-password"
   PGHOST = "18.190.15.34"
   PGPORT = "5432"
   PGDATABASE = "your_database"
   PGUSER = "your_user"
   PGPASSWORD = "your_db_password"
   PGSSLMODE = "require"
   ```

4. Click **Deploy**. In a minute or two you get a public URL like
   `https://your-app.streamlit.app`.

## Step 3 — Open the database to the cloud app

In the AWS console (EC2 -> Security Groups -> the group on your instance ->
Inbound rules), add a rule allowing **PostgreSQL / 5432**. Because Streamlit's
IPs aren't fixed, the source usually has to be broad (`0.0.0.0/0`). This is why
the strong password + SSL + read-only user above matter.

## Step 4 — Test

Open your `.streamlit.app` URL. You should see the password box, then the app.
If it can't reach the database, the sidebar shows the exact error.

## Updating later

Re-upload changed files to the repo (or `git push`), and Streamlit Cloud
redeploys automatically.
