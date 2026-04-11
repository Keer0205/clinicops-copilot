"""
ChromaDB retrieval and answer-generation logic.
Shared by app.py and scripts/ci_eval.py.
"""

from __future__ import annotations

from typing import Any

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from utils.embed import embed_query, embed_texts, DEFAULT_EMBED_MODEL
from utils.pdf import pdf_to_chunks

REFUSAL_TEXT = "I couldn't find that in the uploaded clinic documents."
DEFAULT_COLLECTION = "clinic_docs"
DEFAULT_TOP_K = 8
DEFAULT_CHAT_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------

def get_chroma_collection(
    chroma_dir: str,
    collection_name: str = DEFAULT_COLLECTION,
    anonymized_telemetry: bool = False,
) -> tuple[chromadb.PersistentClient, chromadb.Collection]:
    """Return (client, collection), creating the collection if it does not exist."""
    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=anonymized_telemetry),
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


def reset_collection(
    chroma_client: chromadb.PersistentClient,
    collection_name: str = DEFAULT_COLLECTION,
) -> chromadb.Collection:
    """Delete and recreate *collection_name*, returning the fresh collection."""
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass
    return chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def upsert_pdf(
    pdf_bytes: bytes,
    source_name: str,
    collection: chromadb.Collection,
    openai_client: OpenAI,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> int:
    """
    Extract, chunk, embed, and upsert a PDF into *collection*.

    Returns the number of chunks indexed.
    """
    records = pdf_to_chunks(pdf_bytes, source_name)
    if not records:
        return 0

    ids = [r["id"] for r in records]
    texts = [r["text"] for r in records]
    metadatas = [{"source": r["source"], "page": r["page"]} for r in records]
    embeddings = embed_texts(texts, openai_client, embed_model)

    collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    return len(records)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    collection: chromadb.Collection,
    openai_client: OpenAI,
    embed_model: str = DEFAULT_EMBED_MODEL,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Embed *query* and return the top-k matching (documents, metadatas).
    """
    q_emb = embed_query(query, openai_client, embed_model)
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas"],
    )
    docs = res["documents"][0] if res.get("documents") else []
    metas = res["metadatas"][0] if res.get("metadatas") else []
    return docs, metas


def build_context(docs: list[str], metas: list[dict[str, Any]]) -> str:
    """Format retrieved passages into a numbered context block for the LLM."""
    blocks = []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        blocks.append(
            f"[{i}] SOURCE: {meta.get('source')} | PAGE: {meta.get('page')}\n{doc}"
        )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def answer_with_citations(
    question: str,
    collection: chromadb.Collection,
    openai_client: OpenAI,
    embed_model: str = DEFAULT_EMBED_MODEL,
    chat_model: str = DEFAULT_CHAT_MODEL,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """
    Retrieve relevant passages and generate a grounded answer.

    Returns::

        {
            "answer":          str,
            "citations":       list[{"source": str, "page": int}],
            "refused":         bool,
            "citations_count": int,
        }

    The function refuses (without calling the LLM) when:
    - No passages are retrieved, or
    - The first retrieved passage is suspiciously short (< 80 chars).

    The *min_sources* check has been intentionally removed: a single
    well-matched document is sufficient to give a valid grounded answer.
    """
    docs, metas = retrieve(question, collection, openai_client, embed_model, top_k)

    # Hard refuse: no content retrieved
    if not docs or not metas:
        return {"answer": REFUSAL_TEXT, "citations": [], "refused": True, "citations_count": 0}

    # Hard refuse: top passage is too short to be meaningful
    if len(docs[0].strip()) < 80:
        return {"answer": REFUSAL_TEXT, "citations": [], "refused": True, "citations_count": 0}

    # Deduplicate citations by (source, page)
    citations: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for meta in metas:
        key = (meta.get("source"), meta.get("page"))
        if key not in seen:
            seen.add(key)
            citations.append({"source": key[0], "page": key[1]})

    context = build_context(docs, metas)
    system_prompt = (
        "You are ClinicOps Copilot. Answer ONLY using the provided context from clinic documents. "
        f'If the context is insufficient, say exactly: "{REFUSAL_TEXT}" '
        "Do not guess. Keep the answer concise and practical."
    )
    user_prompt = f"QUESTION:\n{question}\n\nCONTEXT (use this only):\n{context}"

    chat_resp = openai_client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    answer_text = chat_resp.choices[0].message.content.strip()

    # If the model itself refused, clear citations
    if REFUSAL_TEXT in answer_text:
        return {"answer": REFUSAL_TEXT, "citations": [], "refused": True, "citations_count": 0}

    return {
        "answer": answer_text,
        "citations": citations[:6],
        "refused": False,
        "citations_count": len(citations[:6]),
    }
