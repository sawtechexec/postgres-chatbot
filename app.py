"""Streamlit chatbot over an AWS Postgres database.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import db
import queries
import rag

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

    st.divider()
    st.header("Mode")
    _rag_ready = rag.index_ready()
    if _rag_ready:
        mode = st.radio(
            "How should questions be answered?",
            ["SQL (structured queries)", "Search (semantic / RAG)"],
            help="SQL turns your question into a database query. "
                 "Search finds the most relevant text across all four tables "
                 "and answers from it.",
        )
        with st.expander("Index contents"):
            try:
                st.dataframe(rag.index_stats(), width="stretch", hide_index=True)
            except Exception:  # noqa: BLE001
                pass
    else:
        mode = "SQL (structured queries)"
        st.caption("Semantic search is off — run ingest.py to build the index.")

# --- Chat history -----------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []


def render_result(entry: dict) -> None:
    st.markdown(f"**{entry['label']}**")
    if entry.get("sql"):
        st.code(entry["sql"], language="sql")
    if entry.get("answer"):
        st.markdown(entry["answer"])
        st.caption("Sources (most similar chunks):")
    df: pd.DataFrame = entry["df"]
    st.dataframe(df, width="stretch")
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
            if mode.startswith("Search"):
                with st.spinner("Searching…"):
                    answer, sources = rag.answer_with_rag(question)
                st.session_state.history.append(
                    {"label": question, "df": sources, "answer": answer}
                )
            else:
                sql, df = queries.ask_with_llm(question)
                st.session_state.history.append(
                    {"label": question, "df": df, "sql": sql}
                )
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not answer that: {exc}")
