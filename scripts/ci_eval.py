import os
import json
import csv
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
import chromadb

REFUSAL_TEXT = "I couldn't find that in the uploaded clinic documents."


def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set.")
        sys.exit(2)

    try:
        from openai import OpenAI  # openai>=1.x
        client = OpenAI(api_key=api_key)

        def embed(texts, model):
            resp = client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]

        def chat(messages, model):
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
            return resp.choices[0].message.content

        return embed, chat

    except Exception:
        import openai  # openai<1.x
        openai.api_key = api_key

        def embed(texts, model):
            resp = openai.Embedding.create(model=model, input=texts)
            return [d["embedding"] for d in resp["data"]]

        def chat(messages, model):
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0,
            )
            return resp["choices"][0]["message"]["content"]

        return embed, chat


def chunk_text(text: str, max_chars: int = 1200):
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + max_chars])
        i += max_chars
    return chunks


def load_pdf_chunks(pdf_path: Path):
    doc = fitz.open(str(pdf_path))
    out = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        txt = page.get_text("text").strip()
        for chunk in chunk_text(txt):
            out.append({"text": chunk, "source": pdf_path.name, "page": page_idx + 1})
    return out


def build_context(hits):
    lines = []
    for h in hits:
        tag = f"[{h['source']} p{h['page']}]"
        lines.append(f"{tag} {h['text']}")
    return "\n\n".join(lines)


def answer_question(question: str, hits, chat_fn, chat_model: str):
    context = build_context(hits)

    system = (
        "You are ClinicOps Copilot. Answer ONLY using the provided clinic document excerpts. "
        f"If the answer is not clearly present, reply exactly: '{REFUSAL_TEXT}'. "
        "Do not guess. Keep the answer short and practical."
    )
    user = f"Question: {question}\n\nClinic document excerpts:\n{context}"

    return chat_fn(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        chat_model,
    )


def load_eval_questions(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("questions", [])
    if not isinstance(data, list):
        raise ValueError("eval_questions.json must be a list (or {questions:[...]})")
    return data


def load_quality_gate(path: Path):
    gate = json.loads(path.read_text(encoding="utf-8"))
    return {
        "min_pass_rate": float(gate.get("min_pass_rate", 0.9)),
        "min_citation_rate_in_docs": float(gate.get("min_citation_rate_in_docs", 0.9)),
        "min_refusal_rate_not_in_docs": float(gate.get("min_refusal_rate_not_in_docs", 0.9)),
    }


def main():
    repo_root = Path(__file__).resolve().parents[1]
    sample_docs_dir = repo_root / "sample_docs"
    eval_q_path = repo_root / "eval_questions.json"
    gate_path = repo_root / "quality_gate.json"
    out_csv = repo_root / "eval_results" / "ci_latest.csv"

    if not sample_docs_dir.exists():
        print("ERROR: sample_docs/ folder not found in repo.")
        sys.exit(2)
    if not eval_q_path.exists():
        print("ERROR: eval_questions.json not found.")
        sys.exit(2)
    if not gate_path.exists():
        print("ERROR: quality_gate.json not found.")
        sys.exit(2)

    embed_fn, chat_fn = get_openai_client()
    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    questions = load_eval_questions(eval_q_path)
    gate = load_quality_gate(gate_path)

    chunks = []
    pdfs = sorted(sample_docs_dir.glob("*.pdf"))
    if not pdfs:
        print("ERROR: No PDFs found inside sample_docs/.")
        sys.exit(2)

    for pdf in pdfs:
        chunks.extend(load_pdf_chunks(pdf))

    chroma_path = repo_root / ".ci_chroma"
    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        client.delete_collection("clinic_docs")
    except Exception:
        pass
    col = client.create_collection("clinic_docs", metadata={"hnsw:space": "cosine"})

    texts = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]
    ids = [f"c{i}" for i in range(len(chunks))]

    embeddings = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        embeddings.extend(embed_fn(texts[i : i + batch_size], embed_model))

    col.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)

    rows = []
    in_docs_total = 0
    in_docs_with_cites = 0
    not_in_docs_total = 0
    not_in_docs_refused = 0
    passed_total = 0

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    for q in questions:
        qid = q.get("id", "")
        qtype = q.get("type", "")
        question = q.get("question", "")

        t0 = time.time()

        q_emb = embed_fn([question], embed_model)[0]
        res = col.query(
            query_embeddings=[q_emb],
            n_results=4,
            include=["documents", "metadatas", "distances"],
        )

        docs = res["documents"][0] if res.get("documents") else []
        metas = res["metadatas"][0] if res.get("metadatas") else []

        hits = [{"text": d, "source": m.get("source", ""), "page": m.get("page", "")} for d, m in zip(docs, metas)]

        answer = answer_question(question, hits, chat_fn, chat_model).strip()
        latency_ms = (time.time() - t0) * 1000.0

        refused = (answer == REFUSAL_TEXT) or (REFUSAL_TEXT.lower() in answer.lower())
        citations_count = 0 if refused else len({(h["source"], h["page"]) for h in hits if h["source"]})

        if qtype == "in_docs":
            in_docs_total += 1
            if citations_count > 0:
                in_docs_with_cites += 1
            passed = (not refused) and (citations_count > 0)
        else:
            not_in_docs_total += 1
            passed = refused
            if refused:
                not_in_docs_refused += 1

        if passed:
            passed_total += 1

        rows.append(
            {
                "id": qid,
                "type": qtype,
                "question": question,
                "passed": passed,
                "refused": refused,
                "citations_count": citations_count,
                "latency_ms": round(latency_ms, 2),
                "answer": answer,
            }
        )

    pass_rate = passed_total / max(1, len(questions))
    citation_rate_in_docs = in_docs_with_cites / max(1, in_docs_total)
    refusal_rate_not_in_docs = not_in_docs_refused / max(1, not_in_docs_total)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["id", "type", "question", "passed", "refused", "citations_count", "latency_ms", "answer"],
        )
        w.writeheader()
        w.writerows(rows)

    print("CI Eval Summary")
    print(f"pass_rate={pass_rate:.3f}")
    print(f"citation_rate_in_docs={citation_rate_in_docs:.3f}")
    print(f"refusal_rate_not_in_docs={refusal_rate_not_in_docs:.3f}")
    print(f"Report saved: {out_csv}")

    failed = []
    if pass_rate < gate["min_pass_rate"]:
        failed.append("pass_rate")
    if citation_rate_in_docs < gate["min_citation_rate_in_docs"]:
        failed.append("citation_rate_in_docs")
    if refusal_rate_not_in_docs < gate["min_refusal_rate_not_in_docs"]:
        failed.append("refusal_rate_not_in_docs")

    if failed:
        print(f"QUALITY GATE FAIL: {', '.join(failed)}")
        sys.exit(1)

    print("QUALITY GATE PASS ✅")


if __name__ == "__main__":
    main()
