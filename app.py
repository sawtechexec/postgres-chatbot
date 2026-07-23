"""Streamlit chatbot over an AWS Postgres database.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import io
from pypdf import PdfReader
import docx

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

def extract_jd_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if name.endswith(".docx"):
        return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
    return data.decode("utf-8", errors="ignore")

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
        st.caption(
            "Questions are answered automatically: content questions use "
            "semantic search; counts, dates, and lookups use SQL."
        )
        with st.expander("Index contents"):
            try:
                st.dataframe(rag.index_stats(), width="stretch", hide_index=True)
            except Exception:  # noqa: BLE001
                pass
    else:
        st.caption("Semantic search is off — run ingest.py to build the index.")

# --- Chat history -----------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []


def _save_feedback(idx: int) -> None:
    entry = st.session_state.history[idx]
    rating = st.session_state.get(f"feedback_{idx}")
    if rating is None:
        return
    try:
        db.log_feedback(
            entry.get("label", ""),
            entry.get("answer", ""),
            "up" if rating == 1 else "down",
        )
    except Exception:  # noqa: BLE001
        pass  # never let feedback logging break the app


def render_result(entry: dict, idx: int) -> None:
    st.markdown(f"**{entry['label']}**")
    if entry.get("answer"):
        st.markdown(entry["answer"])
    df: pd.DataFrame = entry["df"]
    with st.expander("Details"):
        if entry.get("sql"):
            st.code(entry["sql"], language="sql")
        st.dataframe(df, width="stretch")
        if df.shape[1] == 2 and pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
            try:
                st.bar_chart(df.set_index(df.columns[0]))
            except Exception:  # noqa: BLE001
                pass
        st.caption(f"{len(df)} rows")
    st.feedback("thumbs", key=f"feedback_{idx}", on_change=_save_feedback, args=(idx,))


for i, entry in enumerate(st.session_state.history):
    with st.chat_message("assistant"):
        render_result(entry, i)

# --- Ask questions in plain English -----------------------------------------
if not queries.openai_available():
    st.info(
        "Natural-language mode is off. Add OPENAI_API_KEY to your .env file "
        "to enable asking questions in plain English."
    )
else:
    jd_file = st.file_uploader(
        "Optional: upload a job description to find matching candidates",
        type=["pdf", "docx", "txt"],
    )
    if jd_file and st.button("Find candidates for this JD"):
        try:
            with st.spinner("Reading JD and searching candidates…"):
                jd_text = extract_jd_text(jd_file)
                if len(jd_text.strip()) < 100:
                    st.warning("Couldn't extract much text — is it a scanned PDF?")
                else:
                    answer, sources = rag.match_candidates_to_jd(jd_text)
                    st.session_state.history.append(
                        {"label": f"JD match: {jd_file.name}", "df": sources, "answer": answer}
                    )
                    st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not process the JD: {exc}")

    question = st.chat_input("Ask a question about your data…")
    if question:
        try:
            with st.spinner("Thinking…"):
                route = rag.route_question(question) if _rag_ready else "sql"
                if route == "hybrid":
                    answer, sources = rag.answer_hybrid(question)
                    entry = {"label": question, "df": sources, "answer": answer}
                    if sources.attrs.get("sql"):
                        entry["sql"] = sources.attrs["sql"]
                    st.session_state.history.append(entry)
                elif route == "search":
                    answer, sources = rag.answer_with_rag(question)
                    st.session_state.history.append(
                        {"label": question, "df": sources, "answer": answer}
                    )
                else:
                    sql, df = queries.ask_with_llm(question)
                    answer = rag.summarize_rows(question, sql, df)
                    st.session_state.history.append(
                        {"label": question, "df": df, "sql": sql, "answer": answer}
                    )
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not answer that: {exc}")
