"""
Embedding utilities — thin wrapper around the OpenAI embeddings API.
Shared by app.py and scripts/ci_eval.py.
"""

from __future__ import annotations

from openai import OpenAI

DEFAULT_EMBED_MODEL = "text-embedding-3-small"
_BATCH_SIZE = 64  # OpenAI recommended max batch size for embeddings


def embed_texts(
    texts: list[str],
    client: OpenAI,
    model: str = DEFAULT_EMBED_MODEL,
) -> list[list[float]]:
    """
    Return embeddings for *texts* using the specified OpenAI *model*.

    Automatically batches requests so that large lists don't exceed API limits.
    """
    if not texts:
        return []

    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        resp = client.embeddings.create(model=model, input=batch)
        all_embeddings.extend(d.embedding for d in resp.data)
    return all_embeddings


def embed_query(
    query: str,
    client: OpenAI,
    model: str = DEFAULT_EMBED_MODEL,
) -> list[float]:
    """Return a single embedding vector for *query*."""
    return embed_texts([query], client, model)[0]
