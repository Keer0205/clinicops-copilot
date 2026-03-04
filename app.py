import os
import re
import time
from typing import List, Dict, Any, Tuple

import streamlit as st
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from openai import OpenAI

# ----------------------------
# Page
# ----------------------------
st.set_page_config(page_title="ClinicOps Copilot", page_icon="🩺", layout="wide")
st.title("🩺 ClinicOps Copilot — Ask My Clinic Docs")
st.caption("Upload clinic PDFs (SOPs, consent forms, aftercare, pricing). Ask questions. Answers include page citations.")

# ----------------------------
# Day 2: session status
# ----------------------------
if "indexed_chunks" not in st.session_state:
    st.session_state.indexed_chunks = 0
if "last_indexed_at" not in st.session_state:
    st.session_state.last_indexed_at = None

status_left, status_right = st.columns(2)
with status_left:
    st.info(f"Indexed chunks: {st.session_state.indexed_chunks}")
with status_right:
    st.info(f"Last indexed: {st.session_state.last_indexed_at or '—'}")

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
        return {"answer": "I couldn’t find that in the uploaded clinic documents.", "citations": []}
    if len(docs[0].strip()) < 80:
       return {"answer": "I couldn’t find that in the uploaded clinic documents.", "citations": [], "refused": True}
        
    citations, seen = [], set()
    for m in metas:
        key = (m.get("source"), m.get("page"))
        if key not in seen:
            seen.add(key)
            citations.append({"source": key[0], "page": key[1]})

    # If too few sources, refuse (and force citations empty)
    if len(citations) < min_sources:
        return {"answer": "I couldn’t find that in the uploaded clinic documents.", "citations": [], "refused": True}

    context = build_context(docs, metas)

    system = (
        "You are ClinicOps Copilot. Answer ONLY using the provided context from clinic documents. "
        "If context is insufficient, say: \"I couldn’t find that in the uploaded clinic documents.\" "
        "Do not guess."
    )

    user = f"QUESTION:\n{question}\n\nCONTEXT (use this only):\n{context}"

    chat = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )

   return {"answer": chat.choices[0].message.content.strip(), "citations": citations[:6], "refused": False}


# ----------------------------
# Sidebar: Upload & Index + Clear DB
# ----------------------------
with st.sidebar:
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
st.subheader("2) Ask a question")
question = st.text_input("Type your question…", placeholder="e.g., What is the cancellation policy?")

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Ask"):
    if not question.strip():
        st.warning("Type a question first.")
    else:
        start = time.time()
        result = answer_with_citations(question.strip())
        ms = (time.time() - start) * 1000
        st.session_state.history.insert(0, {"q": question, "result": result, "ms": ms})

for item in st.session_state.history[:10]:
    st.markdown(f"### Q: {item['q']}")
    st.write(item["result"]["answer"])
    st.caption(f"Latency: {item['ms']:.0f} ms")

    if item["result"]["citations"]:
        st.markdown("**Citations:**")
        for c in item["result"]["citations"]:
            st.write(f"- {c['source']} p.{c['page']}")
    st.divider()
