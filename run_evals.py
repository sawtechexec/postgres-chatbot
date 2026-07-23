#!/usr/bin/env python3
"""
Eval harness for the Tech Exec chatbot.

Run on EC2:
    python3 ~/rag/run_evals.py ~/rag/eval_set.jsonl
    python3 ~/rag/run_evals.py ~/rag/eval_set.jsonl --ids eval_001   # smoke test

Reads a JSONL eval set:
    {"id": "eval_001", "question": "...", "expected": ["element 1", ...]}

For each question it runs the same pipeline as the deployed Streamlit app
(route -> search / rag / hybrid), judges the answer against the expected
elements with a cheap LLM call, and logs full detail (including retrieved
context) so failures can be diagnosed as retrieval vs. answering problems.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

# --- environment & imports from the deployed repo ---------------------------

REPO_DIR = os.path.expanduser("~/postgres-chatbot")
ENV_FILE = os.path.expanduser("~/rag/.env")

sys.path.insert(0, REPO_DIR)

from dotenv import load_dotenv

load_dotenv(ENV_FILE)  # DB creds (172.31.42.61) + OPENAI_API_KEY

import pandas as pd  # noqa: E402
from openai import OpenAI  # noqa: E402

import queries  # noqa: E402  (from ~/postgres-chatbot)
import rag  # noqa: E402      (from ~/postgres-chatbot)

judge_client = OpenAI()
JUDGE_MODEL = "gpt-4o"

# --- pipeline adapter: mirrors app.py lines 119-133 --------------------------


def _df_to_context(df: "pd.DataFrame") -> str:
    """Render retrieved chunk rows for the log."""
    if df is None or df.empty:
        return "(no sources retrieved)"
    lines = []
    for i, r in enumerate(df.itertuples()):
        table = getattr(r, "source_table", "?")
        sid = getattr(r, "source_id", "?")
        content = str(getattr(r, "content", ""))[:1500]
        lines.append(f"[{i}] {table} #{sid}\n{content}")
    return "\n\n---\n\n".join(lines)


def run_pipeline(question: str):
    """Returns (answer_text, retrieved_context_text, route)."""
    route = rag.route_question(question)
    if route == "hybrid":
        answer, sources = rag.answer_hybrid(question)
        context = _df_to_context(sources)
    elif route == "search":
        answer, sources = rag.answer_with_rag(question)
        context = _df_to_context(sources)
    else:  # sql
        sql, df = queries.ask_with_llm(question)
        answer = rag.summarize_rows(question, sql, df)
        context = f"SQL: {sql}\nROWS (first 50):\n{df.head(50).to_csv(index=False)}"
    return answer, context, route


# --- judge -------------------------------------------------------------------

JUDGE_PROMPT = """You are grading a recruiting-database chatbot's answer.

QUESTION:
{question}

EXPECTED ELEMENTS (a correct answer should contain each of these, unless the \
element itself allows "Not available" / "No evidence found" and the answer \
correctly says so):
{expected}

CHATBOT ANSWER:
{answer}

Grade the answer. Rules:
- An element counts as covered if the SUBSTANCE is present anywhere in the \
answer, even if not labeled with the element's exact words. For explanation/\
reasoning elements, any concrete supporting reasoning counts.
- An element also counts as covered if the answer correctly states the \
information is not available in the data (when acceptable per the element \
description).
- Before marking an element missing, quote to yourself the part of the answer \
that would cover it; mark it missing ONLY if no such part exists. Be \
reasonable, not pedantic.
- Respond ONLY with JSON, no markdown fences, exactly this shape:
{{"verdict": "PASS" or "FAIL", "covered": ["..."], "missing": ["..."], "note": "one sentence"}}
- PASS means every expected element is covered. Anything missing = FAIL.
"""


def judge(question, expected, answer):
    prompt = JUDGE_PROMPT.format(
        question=question,
        expected="\n".join(f"- {e}" for e in expected),
        answer=answer,
    )
    resp = judge_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"verdict": "FAIL", "covered": [], "missing": expected,
                "note": f"Judge returned unparseable output: {raw[:200]}"}


# --- main loop ---------------------------------------------------------------


def load_eval_set(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_file")
    ap.add_argument("--ids", help="comma-separated eval ids to run (default: all)")
    args = ap.parse_args()

    items = load_eval_set(args.eval_file)
    if args.ids:
        wanted = set(args.ids.split(","))
        items = [it for it in items if it["id"] in wanted]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.expanduser(f"~/rag/eval_results_{ts}.jsonl")
    results = []

    print(f"Running {len(items)} evals -> {out_path}\n")

    with open(out_path, "w") as out:
        for it in items:
            qid, question, expected = it["id"], it["question"], it["expected"]
            print(f"{qid}: {question[:70]}...", flush=True)
            t0 = time.time()
            try:
                answer, context, route = run_pipeline(question)
                error = None
            except Exception as e:  # noqa: BLE001
                answer, context, route, error = "", "", "error", repr(e)

            if error:
                verdict = {"verdict": "ERROR", "covered": [], "missing": expected,
                           "note": error}
            else:
                verdict = judge(question, expected, answer)

            rec = {
                "id": qid,
                "question": question,
                "route": route,
                "answer": answer,
                "retrieved_context": context,
                "expected": expected,
                "verdict": verdict["verdict"],
                "missing": verdict.get("missing", []),
                "judge_note": verdict.get("note", ""),
                "seconds": round(time.time() - t0, 1),
            }
            results.append(rec)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"    -> [{route}] {verdict['verdict']}"
                  + (f" (missing: {', '.join(verdict.get('missing', []))[:80]})"
                     if verdict["verdict"] != "PASS" else ""))

    n = len(results)
    passed = sum(1 for r in results if r["verdict"] == "PASS")
    failed = sum(1 for r in results if r["verdict"] == "FAIL")
    errored = sum(1 for r in results if r["verdict"] == "ERROR")
    print("\n" + "=" * 60)
    print(f"PASS {passed}/{n}   FAIL {failed}   ERROR {errored}   "
          f"score {100.0 * passed / max(n, 1):.0f}%")
    print(f"Details: {out_path}")
    print("=" * 60)
    if failed or errored:
        print("\nFailures by id:")
        for r in results:
            if r["verdict"] != "PASS":
                print(f"  {r['id']} [{r['route']}]: {r['judge_note'][:100]}")


if __name__ == "__main__":
    main()
