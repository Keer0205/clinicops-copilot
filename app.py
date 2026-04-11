"""
ClinicOps Copilot — main Streamlit application.

Responsibilities (UI only):
- File upload and indexing
- Session state management
- Question input and answer display
- Session monitoring metrics
- Evaluation mode (in-app)

All heavy logic lives in utils/ so it can be reused by CI scripts.
"""

from __future__ import annotations

import csv
import io
import json
import os
import time

import pandas as pd
import streamlit as st
from openai import OpenAI

from utils.retrieval import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    answer_with_citations,
    get_chroma_collection,
    reset_collection,
    upsert_pdf,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ClinicOps Copilot", page_icon="🩺", layout="wide")
st.title("🩺 ClinicOps Copilot — Ask My Clinic Docs")
st.caption(
    "Upload clinic PDFs (SOPs, consent forms, aftercare, pricing). "
    "Ask questions. Answers include page citations."
)
st.success("✅ Demo ready: Upload clinic PDFs and ask questions. Answers include page citations.")
st.caption(
    "Disclaimer: This assistant answers ONLY from the uploaded clinic documents. "
    "If it's not in the PDFs, it will refuse."
)

# ---------------------------------------------------------------------------
# OpenAI client (secrets only — never expose the key in the UI)
# ---------------------------------------------------------------------------
api_key = st.secrets.get("OPENAI_API_KEY", None)
if not api_key:
    st.error(
        "OPENAI_API_KEY missing. "
        "Add it via Streamlit Cloud → App → Settings → Secrets, "
        "or create a .streamlit/secrets.toml locally (see .secrets.toml.example)."
    )
    st.stop()

openai_client = OpenAI(api_key=api_key.strip())
embed_model = st.secrets.get("OPENAI_EMBED_MODEL", DEFAULT_EMBED_MODEL)
chat_model = st.secrets.get("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------
CHROMA_DIR = os.path.join(os.getcwd(), "data", "chroma")
os.makedirs(CHROMA_DIR, exist_ok=True)
chroma_client, collection = get_chroma_collection(CHROMA_DIR)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "indexed_chunks" not in st.session_state:
    st.session_state.indexed_chunks = 0
if "last_indexed_at" not in st.session_state:
    st.session_state.last_indexed_at = None
if "history" not in st.session_state:
    st.session_state.history = []
if "eval_results" not in st.session_state:
    st.session_state.eval_results = []

# ---------------------------------------------------------------------------
# Status bar
# ---------------------------------------------------------------------------
col_left, col_right = st.columns(2)
with col_left:
    st.info(f"Indexed chunks: {st.session_state.indexed_chunks}")
with col_right:
    st.info(f"Last indexed: {st.session_state.last_indexed_at or '—'}")

# ---------------------------------------------------------------------------
# Monitoring metrics (rolling last 20 questions)
# ---------------------------------------------------------------------------
def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] if f == c else xs[f] + (xs[c] - xs[f]) * (k - f)


_recent = st.session_state.history[:20]
latencies = [x["ms"] for x in _recent if isinstance(x.get("ms"), (int, float))]
total_q = len(st.session_state.history)
refused_n = sum(1 for x in st.session_state.history if x.get("refused"))
refuse_rate = (refused_n / total_q * 100.0) if total_q else 0.0
p50 = _percentile(latencies, 50)
p95 = _percentile(latencies, 95)

m1, m2, m3 = st.columns(3)
m1.metric("Questions (this session)", str(total_q))
m2.metric("Refusal rate", f"{refuse_rate:.0f}%")
m3.metric("Latency p50 / p95", f"{(p50 or 0):.0f} / {(p95 or 0):.0f} ms")

# ---------------------------------------------------------------------------
# CSV log export
# ---------------------------------------------------------------------------
def _history_to_csv(rows: list[dict]) -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["timestamp", "question", "latency_ms", "refused", "citations_count"])
    for r in rows:
        writer.writerow([
            r.get("ts", ""),
            r.get("q", ""),
            str(int(float(r.get("ms", 0)))),
            str(bool(r.get("refused", False))),
            str(int(r.get("citations_count", 0))),
        ])
    return out.getvalue()


st.download_button(
    "Download session logs (CSV)",
    data=_history_to_csv(st.session_state.history),
    file_name="clinicops_session_logs.csv",
    mime="text/csv",
)

st.divider()

# ---------------------------------------------------------------------------
# Sidebar: upload & index
# ---------------------------------------------------------------------------
with st.sidebar:
    st.info("Tip: Upload the 3 separate sample PDFs (not a combined pack) for cleaner citations.")
    st.header("1) Upload clinic PDFs")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    col_a, col_b = st.columns(2)
    do_index = col_a.button("Index documents", use_container_width=True)
    do_clear = col_b.button("Clear DB", use_container_width=True)

    if do_clear:
        collection = reset_collection(chroma_client)
        st.session_state.indexed_chunks = 0
        st.session_state.last_indexed_at = None
        st.session_state.history = []
        st.success("Cleared database ✅")

    if uploaded_files and do_index:
        with st.spinner("Indexing PDFs into the knowledge base…"):
            total_chunks = 0
            for f in uploaded_files:
                total_chunks += upsert_pdf(
                    f.getvalue(),
                    source_name=f.name,
                    collection=collection,
                    openai_client=openai_client,
                    embed_model=embed_model,
                )
            st.session_state.indexed_chunks = total_chunks
            st.session_state.last_indexed_at = time.strftime("%Y-%m-%d %H:%M:%S")
            st.success(f"Indexed {total_chunks} chunks ✅")

# ---------------------------------------------------------------------------
# Eval mode
# ---------------------------------------------------------------------------
st.subheader("✅ Eval mode")
col_e1, col_e2, col_e3 = st.columns([1, 1, 2])
with col_e1:
    run_eval = st.button("Run eval")
with col_e2:
    clear_eval = st.button("Clear eval results")
with col_e3:
    st.caption("Runs questions from eval_questions.json and reports citation/refusal rates.")

if clear_eval:
    st.session_state.eval_results = []
    st.success("Eval results cleared.")


def _load_eval_questions(path: str = "eval_questions.json") -> list[dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else data.get("questions", [])
    except Exception as exc:
        st.error(f"Could not load {path}: {exc}")
        return []


if run_eval:
    qs = _load_eval_questions()
    if qs:
        st.info(f"Running eval on {len(qs)} questions…")
        results: list[dict] = []
        for q in qs:
            question_text = (q.get("question", "") if isinstance(q, dict) else str(q)).strip()
            if not question_text:
                continue
            q_id = q.get("id", "") if isinstance(q, dict) else ""
            q_type = q.get("type", "in_docs") if isinstance(q, dict) else "in_docs"

            t0 = time.time()
            r = answer_with_citations(
                question_text, collection, openai_client, embed_model, chat_model
            )
            ms = (time.time() - t0) * 1000.0

            refused = bool(r.get("refused", False))
            cites = int(r.get("citations_count", 0))
            passed = (not refused and cites > 0) if q_type == "in_docs" else refused

            results.append({
                "id": q_id,
                "type": q_type,
                "question": question_text,
                "passed": passed,
                "refused": refused,
                "citations_count": cites,
                "latency_ms": round(ms, 2),
                "answer": r.get("answer", ""),
            })

        st.session_state.eval_results = results
        st.success("Eval completed.")

if st.session_state.eval_results:
    df = pd.DataFrame(st.session_state.eval_results)
    total = len(df)
    passed = int(df["passed"].sum())
    in_docs = df[df["type"] == "in_docs"]
    not_in_docs = df[df["type"] == "not_in_docs"]

    pass_rate = (passed / total * 100.0) if total else 0.0
    citation_rate = (
        in_docs["citations_count"].gt(0).sum() / len(in_docs) * 100.0
        if len(in_docs) > 0 else 0.0
    )
    refusal_rate_ood = (
        not_in_docs["refused"].sum() / len(not_in_docs) * 100.0
        if len(not_in_docs) > 0 else 0.0
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Pass rate", f"{pass_rate:.0f}%")
    c2.metric("Citation rate (in_docs)", f"{citation_rate:.0f}%")
    c3.metric("Refusal rate (not_in_docs)", f"{refusal_rate_ood:.0f}%")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download eval report (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="eval_report.csv",
        mime="text/csv",
    )

st.divider()

# ---------------------------------------------------------------------------
# Q&A
# ---------------------------------------------------------------------------
st.subheader("2) Ask a question")
st.markdown("**Try example questions:**")

examples = [
    "What is the cancellation policy?",
    "Is a patch test required for laser hair removal?",
    "What should be avoided after a chemical peel for 7 days?",
    "When can makeup be applied after microneedling?",
    "What should be avoided after Botox?",
    "What are urgent warning signs after dermal fillers?",
]
q_cols = st.columns(3)
for i, ex in enumerate(examples):
    if q_cols[i % 3].button(ex, use_container_width=True):
        st.session_state["prefill_q"] = ex

question = st.text_input(
    "Type your question…",
    value=st.session_state.get("prefill_q", ""),
    placeholder="e.g., What is the cancellation policy?",
)

if st.button("Ask"):
    if not question.strip():
        st.warning("Type a question first.")
    else:
        # Clear prefill so the box is empty on next interaction
        st.session_state.pop("prefill_q", None)

        with st.spinner("Searching clinic documents…"):
            t0 = time.time()
            result = answer_with_citations(
                question.strip(), collection, openai_client, embed_model, chat_model
            )
            ms = (time.time() - t0) * 1000.0

        # Rolling window — keep at most 50 entries to avoid session bloat
        st.session_state.history.insert(0, {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "q": question,
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
            "refused": bool(result.get("refused", False)),
            "citations_count": int(result.get("citations_count", 0)),
            "ms": float(ms),
        })
        st.session_state.history = st.session_state.history[:50]

# ---------------------------------------------------------------------------
# History display (last 10)
# ---------------------------------------------------------------------------
for item in st.session_state.history[:10]:
    st.markdown(f"### Q: {item.get('q', '')}")
    st.caption(f"Time: {item.get('ts', '')}")
    st.write(item.get("answer", ""))
    st.caption(f"Latency: {float(item.get('ms', 0)):.0f} ms")

    cites = item.get("citations", [])
    if not item.get("refused") and cites:
        with st.expander(f"📄 {len(cites)} source(s) cited"):
            for c in cites:
                st.markdown(f"- **{c.get('source')}** — page {c.get('page')}")
    st.divider()
