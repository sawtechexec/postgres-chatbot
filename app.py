"""Streamlit chatbot over an AWS Postgres database.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import db
import queries

st.set_page_config(page_title="Postgres Chatbot", page_icon="🗄️", layout="wide")


def check_password() -> None:
    """Gate the whole app behind a password (set via APP_PASSWORD secret).

    If no APP_PASSWORD is configured, the gate is skipped (e.g. local dev).
    """
    expected = db.get_setting("APP_PASSWORD")
    if not expected:
        return  # no password configured -> open (fine for localhost)

    if st.session_state.get("auth_ok"):
        return

    st.title("🔒 Sign in")
    entered = st.text_input("Password", type="password")
    if st.button("Enter"):
        if entered == str(expected):
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


check_password()

st.title("🗄️ Postgres Data Chatbot")

# --- Connection status ------------------------------------------------------
with st.sidebar:
    st.header("Connection")
    ok, msg = db.test_connection()
    if ok:
        st.success("Connected")
        st.caption(msg)
    else:
        st.error("Not connected")
        st.caption(msg)
        st.stop()

    st.divider()
    st.header("Schema")
    try:
        tables_df = db.list_tables()
        tables_df["full"] = tables_df["table_schema"] + "." + tables_df["table_name"]
        st.caption(f"{len(tables_df)} tables found")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read schema: {exc}")
        st.stop()

# --- Chat history -----------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []


def render_result(entry: dict) -> None:
    st.markdown(f"**{entry['label']}**")
    if entry.get("sql"):
        st.code(entry["sql"], language="sql")
    df: pd.DataFrame = entry["df"]
    st.dataframe(df, use_container_width=True)
    # Offer a quick chart when it makes sense.
    if df.shape[1] == 2 and pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
        try:
            st.bar_chart(df.set_index(df.columns[0]))
        except Exception:  # noqa: BLE001
            pass
    st.caption(f"{len(df)} rows")


for entry in st.session_state.history:
    with st.chat_message("assistant"):
        render_result(entry)

# --- Ask questions in plain English -----------------------------------------
if not queries.openai_available():
    st.info(
        "Natural-language mode is off. Add OPENAI_API_KEY to your .env file "
        "to enable asking questions in plain English."
    )
else:
    question = st.chat_input("Ask a question about your data…")
    if question:
        try:
            sql, df = queries.ask_with_llm(question)
            st.session_state.history.append(
                {"label": question, "df": df, "sql": sql}
            )
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not answer that: {exc}")
