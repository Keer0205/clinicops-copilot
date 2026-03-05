import os
import re
import time
import csv
import io
from typing import List, Dict, Any, Tuple

import streamlit as st
import streamlit.components.v1 as components
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from openai import OpenAI

REFUSAL_TEXT = "I couldn’t find that in the uploaded clinic documents."

# ----------------------------
# Page
# ----------------------------
st.set_page_config(page_title="ClinicOps Copilot", page_icon="🩺", layout="wide")
st.title("🩺 ClinicOps Copilot — Ask My Clinic Docs")
st.caption("Upload clinic PDFs (SOPs, consent forms, aftercare, pricing). Ask questions. Answers include page citations.")
st.success("✅ Demo ready: Upload clinic PDFs and ask questions. Answers include page citations.")
st.caption("Disclaimer: This assistant answers ONLY from the uploaded clinic documents. If it’s not in the PDFs, it will refuse.")
st.caption("Want this for your clinic? DM **CLINICOPS** for a setup call + trial.")
# ----------------------------
# Day 7: Shareable demo pitch (copy-friendly)
# ----------------------------
st.subheader("Share this demo")

# Put your public Streamlit URL here once, then it stays (optional)
default_url = st.secrets.get("APP_URL", "")
demo_url = st.text_input("Demo URL (optional)", value=default_url, placeholder="https://<your-app>.streamlit.app")

pitch = f"""Hi Doctor 👋
I built a small assistant for clinics: **ClinicOps Copilot**.

✅ Upload your clinic PDFs (SOPs, consent, aftercare, pricing)
✅ Staff can ask questions and get answers with **page citations**
✅ If it’s not in your documents, it **refuses** (no guessing)
✅ Basic monitoring: p50/p95 latency + downloadable CSV logs

Demo link: {demo_url if demo_url else "[paste demo link here]"}
If you want, I can set this up for your clinic for a quick trial."""

st.text_area("Copy message (WhatsApp/Email)", value=pitch, height=220)
safe_pitch = pitch.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
components.html(
    f"""
    <button id="copybtn" style="padding:8px 12px; border-radius:8px; border:1px solid #ddd; cursor:pointer;">
      Copy message to clipboard
    </button>
    <span id="copystatus" style="margin-left:10px; color:#2e7d32;"></span>

    <textarea id="pitch" style="position:absolute; left:-9999px;">{safe_pitch}</textarea>

    <script>
      const btn = document.getElementById("copybtn");
      const txt = document.getElementById("pitch");
      const status = document.getElementById("copystatus");
      btn.onclick = async () => {{
        try {{
          await navigator.clipboard.writeText(txt.value);
          status.textContent = "Copied ✅";
          setTimeout(()=>status.textContent="", 2000);
        }} catch (e) {{
          status.textContent = "Copy failed — select & copy manually.";
        }}
      }}
    </script>
    """,
    height=60,
)

st.caption("Tip: Add APP_URL in Streamlit Secrets to prefill the demo link.")
# ----------------------------
# Day 2: session status
# ----------------------------
if "indexed_chunks" not in st.session_state:
    st.session_state.indexed_chunks = 0
if "last_indexed_at" not in st.session_state:
    st.session_state.last_indexed_at = None
if "history" not in st.session_state:
    st.session_state.history = []

status_left, status_right = st.columns(2)
with status_left:
    st.info(f"Indexed chunks: {st.session_state.indexed_chunks}")
with status_right:
    st.info(f"Last indexed: {st.session_state.last_indexed_at or '—'}")
    # ----------------------------
# Day 6: Monitoring (session metrics)
# ----------------------------
def _percentile(values, p):
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)

last_n = st.session_state.history[:20]
latencies = [x.get("ms") for x in last_n if isinstance(x.get("ms"), (int, float))]

total_q = len(st.session_state.history)
refused_n = sum(1 for x in st.session_state.history if x.get("refused"))
refuse_rate = (refused_n / total_q * 100.0) if total_q else 0.0

p50 = _percentile(latencies, 50)
p95 = _percentile(latencies, 95)

m1, m2, m3 = st.columns(3)
m1.metric("Questions (this session)", f"{total_q}")
m2.metric("Refusal rate", f"{refuse_rate:.0f}%")
m3.metric("Latency p50 / p95", f"{(p50 or 0):.0f} / {(p95 or 0):.0f} ms")
# ----------------------------
# Day 6: Export session logs (CSV)
# ----------------------------
def _history_to_csv(rows):
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

csv_data = _history_to_csv(st.session_state.history)
st.download_button(
    "Download logs (CSV)",
    data=csv_data,
    file_name="clinicops_session_logs.csv",
    mime="text/csv",
)

# ----------------------------
# OpenAI key (Secrets only)
# ----------------------------
api_key = st.secrets.get("OPENAI_API_KEY", None)
if not api_key:
    st.error("OPENAI_API_KEY missing. Add it in Streamlit Cloud → App → Settings → Secrets.")
    st.stop()

api_key = api_key.strip().replace("\n", "").replace("\r", "")
client = OpenAI(api_key=api_key)

# ----------------------------
# Chroma local persistent db
# ----------------------------
CHROMA_DIR = os.path.join(os.getcwd(), "data", "chroma")
os.makedirs(CHROMA_DIR, exist_ok=True)

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False),
)

COLLECTION_NAME = "clinic_docs"
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def extract_pdf_pages_from_bytes(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i in range(len(doc)):
        txt = clean_text(doc[i].get_text("text"))
        if txt:
            pages.append({"page_num": i + 1, "text": txt})
    doc.close()
    return pages


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def upsert_pdf_bytes(pdf_bytes: bytes, source_name: str) -> int:
    pages = extract_pdf_pages_from_bytes(pdf_bytes)
    ids, docs, metas = [], [], []

    for p in pages:
        page_num = p["page_num"]
        for idx, chunk in enumerate(chunk_text(p["text"])):
            ids.append(f"{source_name}|p{page_num}|c{idx}")
            docs.append(chunk)
            metas.append({"source": source_name, "page": page_num})

    if not docs:
        return 0

    embeddings = embed_texts(docs)
    collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
    return len(docs)


def retrieve(query: str, k: int = 8) -> Tuple[List[str], List[Dict[str, Any]]]:
    q_emb = embed_texts([query])[0]
    res = collection.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas"])
    return res["documents"][0], res["metadatas"][0]


def build_context(docs: List[str], metas: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, (d, m) in enumerate(zip(docs, metas), start=1):
        blocks.append(f"[{i}] SOURCE: {m.get('source')} | PAGE: {m.get('page')}\n{d}")
    return "\n\n".join(blocks)


def answer_with_citations(question: str, min_sources: int = 2) -> Dict[str, Any]:
    docs, metas = retrieve(question, k=8)

    # Hard refuse if retrieval is empty/weak
    if not docs or not metas:
        return {"answer": REFUSAL_TEXT, "citations": [], "refused": True}
    if len(docs[0].strip()) < 80:
        return {"answer": REFUSAL_TEXT, "citations": [], "refused": True}

    citations, seen = [], set()
    for m in metas:
        key = (m.get("source"), m.get("page"))
        if key not in seen:
            seen.add(key)
            citations.append({"source": key[0], "page": key[1]})

    if len(citations) < min_sources:
        return {"answer": REFUSAL_TEXT, "citations": [], "refused": True}

    context = build_context(docs, metas)

    system = (
        "You are ClinicOps Copilot. Answer ONLY using the provided context from clinic documents. "
        f"If context is insufficient, say: \"{REFUSAL_TEXT}\" "
        "Do not guess."
    )

    user = f"QUESTION:\n{question}\n\nCONTEXT (use this only):\n{context}"

    chat = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )

    answer_text = chat.choices[0].message.content.strip()

    # If model refused, force no citations
    if REFUSAL_TEXT in answer_text:
        return {"answer": REFUSAL_TEXT, "citations": [], "refused": True}

    return {"answer": answer_text, "citations": citations[:6], "refused": False}


# ----------------------------
# Sidebar: Upload & Index + Clear DB
# ----------------------------
with st.sidebar:
    st.info("Tip: Upload only the 3 separate PDFs (not the combined pack) for cleaner citations.")
    st.header("1) Upload clinic PDFs")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    col_a, col_b = st.columns(2)
    do_index = col_a.button("Index documents", use_container_width=True)
    do_clear = col_b.button("Clear DB", use_container_width=True)

    if do_clear:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        st.session_state.indexed_chunks = 0
        st.session_state.last_indexed_at = None
        st.session_state.history = []
        st.success("Cleared database ✅")

    if uploaded_files and do_index:
        with st.spinner("Indexing PDFs into the knowledge base..."):
            total_chunks = 0
            for f in uploaded_files:
                total_chunks += upsert_pdf_bytes(f.getvalue(), source_name=f.name)

            st.session_state.indexed_chunks = total_chunks
            st.session_state.last_indexed_at = time.strftime("%Y-%m-%d %H:%M:%S")
            st.success(f"Indexed approx {total_chunks} chunks ✅")

st.divider()

# ----------------------------
# Main: Ask questions
# ----------------------------
# ------------------------------
# Day 8: Evaluation mode (run eval_questions.json)
# ------------------------------
import json

st.subheader("✅ Eval mode (Day 8)")

col_e1, col_e2, col_e3 = st.columns([1, 1, 2])
with col_e1:
    run_eval = st.button("Run eval (30 Qs)")
with col_e2:
    clear_eval = st.button("Clear eval results")
with col_e3:
    st.caption("Runs questions from eval_questions.json and reports citation/refusal rates.")

if "eval_results" not in st.session_state:
    st.session_state.eval_results = []

if clear_eval:
    st.session_state.eval_results = []
    st.success("Eval results cleared.")

def load_eval_questions(path="eval_questions.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Could not load {path}: {e}")
        return []

if run_eval:
    qs = load_eval_questions()
    if not qs:
        st.stop()

    st.info(f"Running eval on {len(qs)} questions…")
    results = []
    for q in qs:
        # q might be a dict {"id":..., "type":..., "question":...} OR just a string
if isinstance(q, str):
    question_text = q.strip()
    q_id = ""
    q_type = ""
else:
    question_text = str(q.get("question", "")).strip()
    q_id = str(q.get("id", ""))
    q_type = str(q.get("type", ""))
        q_type = q.get("type", "in_docs")
        if not question_text:
            continue

       
       for q in qs:
    if isinstance(q, str):
        question_text = q.strip()
        q_id = ""
        q_type = ""
    else:
        question_text = str(q.get("question", "")).strip()
        q_id = str(q.get("id", ""))
        q_type = str(q.get("type", ""))

    if not question_text:
        continue

    start = time.time()
    r = answer_with_citations(question_text)
    ms = (time.time() - start) * 1000

    results.append({
        "id": q_id,
        "type": q_type,
        "question": question_text,
        "answer": r.get("answer", ""),
        "refused": bool(r.get("refused", False)),
        "citations_count": len(r.get("citations", []) or []),
        "ms": float(ms),
    })
        ms = (time.time() - start) * 1000

        citations_count = len(r.get("citations", []) or [])
        refused = bool(r.get("refused", False))

        # scoring rules
        if qtype == "in_docs":
            passed = (not refused) and (citations_count > 0)
        else:  # not_in_docs
            passed = refused

        results.append({
            "id": q.get("id", ""),
            "type": qtype,
            "question": question_text,
            "passed": passed,
            "refused": refused,
            "citations_count": citations_count,
            "latency_ms": float(ms),
        })

    st.session_state.eval_results = results
    st.success("Eval completed.")

# Show eval summary + table
if st.session_state.eval_results:
    df = pd.DataFrame(st.session_state.eval_results)

    total = len(df)
    passed = int(df["passed"].sum())
    pass_rate = (passed / total * 100.0) if total else 0.0

    in_docs = df[df["type"] == "in_docs"]
    not_in_docs = df[df["type"] == "not_in_docs"]

    citation_rate = 0.0
    if len(in_docs) > 0:
        citation_rate = (in_docs["citations_count"].gt(0).sum() / len(in_docs) * 100.0)

    refusal_rate_not_in_docs = 0.0
    if len(not_in_docs) > 0:
        refusal_rate_not_in_docs = (not_in_docs["refused"].sum() / len(not_in_docs) * 100.0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Pass rate", f"{pass_rate:.0f}%")
    c2.metric("Citation rate (in_docs)", f"{citation_rate:.0f}%")
    c3.metric("Refusal rate (not_in_docs)", f"{refusal_rate_not_in_docs:.0f}%")

    st.dataframe(df, use_container_width=True)

    # Optional download
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download eval report (CSV)", data=csv_bytes, file_name="eval_report.csv", mime="text/csv")
st.subheader("2) Ask a question")
st.markdown("**Try example questions:**")
qcols = st.columns(3)

examples = [
    "What is the cancellation policy?",
    "Is a patch test required for laser hair removal?",
    "What should be avoided after a chemical peel for 7 days?",
    "When can makeup be applied after microneedling?",
    "What should be avoided after Botox?",
    "What are urgent warning signs after dermal fillers?"
]

for i, ex in enumerate(examples):
    if qcols[i % 3].button(ex, use_container_width=True):
        st.session_state["prefill_q"] = ex

question = st.text_input(
    "Type your question…",
    value=st.session_state.get("prefill_q", ""),
    placeholder="e.g., What is the cancellation policy?"
)

if st.button("Ask"):
    if not question.strip():
        st.warning("Type a question first.")
    else:
        start = time.time()
        result = answer_with_citations(question.strip())
        ms = (time.time() - start) * 1000
        st.session_state.history.insert(0, {
    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    "q": question,
    "answer": result.get("answer", ""),
    "refused": bool(result.get("refused", False)),
    "citations_count": len(result.get("citations", []) or []),
    "ms": float(ms),
})

for item in st.session_state.history[:10]:
    st.markdown(f"### Q: {item.get('q','')}")
    st.caption(f"time: {item.get('ts','')}")
    st.write(item.get("answer", ""))
    st.caption(f"Latency: {float(item.get('ms', 0)):.0f} ms")

    if not item.get("refused", False) and item.get("citations_count", 0) > 0:
        st.caption(f"Citations: {item.get('citations_count', 0)} sources")
    st.divider()
