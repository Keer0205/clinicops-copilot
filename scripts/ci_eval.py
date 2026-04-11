"""
CI quality-gate evaluation script.

Loads sample PDFs, indexes them into an ephemeral ChromaDB instance,
runs eval_questions.json, and exits non-zero if any quality threshold is missed.

Uses the same utils/ modules as app.py to guarantee consistent behaviour.

Usage:
    OPENAI_API_KEY=sk-... python scripts/ci_eval.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the repo root importable so `from utils.xxx import ...` works when
# this script is executed from any directory.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import chromadb
from openai import OpenAI

from utils.embed import embed_texts, DEFAULT_EMBED_MODEL
from utils.pdf import pdf_to_chunks
from utils.retrieval import (
    DEFAULT_CHAT_MODEL,
    answer_with_citations,
    reset_collection,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SAMPLE_DOCS_DIR = REPO_ROOT / "sample_docs"
EVAL_Q_PATH = REPO_ROOT / "eval_questions.json"
GATE_PATH = REPO_ROOT / "quality_gate.json"
OUT_CSV = REPO_ROOT / "eval_results" / "ci_latest.csv"
CHROMA_DIR = REPO_ROOT / ".ci_chroma"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_eval_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("questions", [])
    if not isinstance(data, list):
        raise ValueError("eval_questions.json must be a JSON array (or {questions: [...]}).")
    return data


def load_quality_gate(path: Path) -> dict:
    gate = json.loads(path.read_text(encoding="utf-8"))
    return {
        "min_pass_rate": float(gate.get("min_pass_rate", 0.9)),
        "min_citation_rate_in_docs": float(gate.get("min_citation_rate_in_docs", 0.9)),
        "min_refusal_rate_not_in_docs": float(gate.get("min_refusal_rate_not_in_docs", 0.9)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Pre-flight checks ---------------------------------------------------
    for label, path in [
        ("sample_docs/", SAMPLE_DOCS_DIR),
        ("eval_questions.json", EVAL_Q_PATH),
        ("quality_gate.json", GATE_PATH),
    ]:
        if not path.exists():
            print(f"ERROR: {label} not found at {path}")
            sys.exit(2)

    pdfs = sorted(SAMPLE_DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"ERROR: No PDFs found in {SAMPLE_DOCS_DIR}")
        sys.exit(2)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        sys.exit(2)

    embed_model = os.getenv("OPENAI_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    chat_model = os.getenv("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)

    openai_client = OpenAI(api_key=api_key)

    # --- Build ephemeral ChromaDB from sample PDFs ---------------------------
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = reset_collection(chroma_client)

    all_chunks: list[dict] = []
    for pdf_path in pdfs:
        all_chunks.extend(pdf_to_chunks(pdf_path.read_bytes(), source_name=pdf_path.name))

    if not all_chunks:
        print("ERROR: No text extracted from sample PDFs.")
        sys.exit(2)

    print(f"Indexing {len(all_chunks)} chunks from {len(pdfs)} PDF(s)...")

    texts = [c["text"] for c in all_chunks]
    metadatas = [{"source": c["source"], "page": c["page"]} for c in all_chunks]
    ids = [c["id"] for c in all_chunks]

    # Embed in batches (handled inside embed_texts)
    embeddings = embed_texts(texts, openai_client, embed_model)
    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    print(f"Indexed {len(all_chunks)} chunks OK")

    # --- Run eval questions --------------------------------------------------
    questions = load_eval_questions(EVAL_Q_PATH)
    gate = load_quality_gate(GATE_PATH)

    rows: list[dict] = []
    in_docs_total = in_docs_cited = not_in_docs_total = not_in_docs_refused = passed_total = 0

    for q in questions:
        qid = q.get("id", "")
        qtype = q.get("type", "in_docs")
        question = q.get("question", "").strip()
        if not question:
            continue

        t0 = time.time()
        r = answer_with_citations(
            question, collection, openai_client, embed_model, chat_model
        )
        latency_ms = (time.time() - t0) * 1000.0

        refused = bool(r.get("refused", False))
        cites = int(r.get("citations_count", 0))

        if qtype == "in_docs":
            in_docs_total += 1
            if cites > 0:
                in_docs_cited += 1
            passed = (not refused) and (cites > 0)
        else:
            not_in_docs_total += 1
            passed = refused
            if refused:
                not_in_docs_refused += 1

        if passed:
            passed_total += 1

        rows.append({
            "id": qid,
            "type": qtype,
            "question": question,
            "passed": passed,
            "refused": refused,
            "citations_count": cites,
            "latency_ms": round(latency_ms, 2),
            "answer": r.get("answer", ""),
        })

    # --- Compute metrics -----------------------------------------------------
    total_qs = max(1, len(rows))
    pass_rate = passed_total / total_qs
    citation_rate_in_docs = in_docs_cited / max(1, in_docs_total)
    refusal_rate_not_in_docs = not_in_docs_refused / max(1, not_in_docs_total)

    # --- Save CSV ------------------------------------------------------------
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["id", "type", "question", "passed", "refused",
                        "citations_count", "latency_ms", "answer"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # --- Print summary -------------------------------------------------------
    print("\n-- CI Eval Summary --")
    print(f"  Total questions     : {total_qs}")
    print(f"  Passed              : {passed_total}")
    print(f"  pass_rate           : {pass_rate:.3f}  (threshold: {gate['min_pass_rate']})")
    print(f"  citation_rate       : {citation_rate_in_docs:.3f}  (threshold: {gate['min_citation_rate_in_docs']})")
    print(f"  refusal_rate_ood    : {refusal_rate_not_in_docs:.3f}  (threshold: {gate['min_refusal_rate_not_in_docs']})")
    print(f"  Report saved to     : {OUT_CSV}")

    # --- Quality gate --------------------------------------------------------
    failures = []
    if pass_rate < gate["min_pass_rate"]:
        failures.append(f"pass_rate={pass_rate:.3f} < {gate['min_pass_rate']}")
    if citation_rate_in_docs < gate["min_citation_rate_in_docs"]:
        failures.append(f"citation_rate_in_docs={citation_rate_in_docs:.3f} < {gate['min_citation_rate_in_docs']}")
    if refusal_rate_not_in_docs < gate["min_refusal_rate_not_in_docs"]:
        failures.append(f"refusal_rate_not_in_docs={refusal_rate_not_in_docs:.3f} < {gate['min_refusal_rate_not_in_docs']}")

    if failures:
        print("\nQUALITY GATE FAIL")
        for f in failures:
            print(f"  -> {f}")
        sys.exit(1)

    print("\nQUALITY GATE PASS")


if __name__ == "__main__":
    main()
