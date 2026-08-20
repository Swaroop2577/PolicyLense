"""
rerank.py

Scores every retrieved chunk against the query using a cross-encoder,
then keeps the top-k most relevant. Cross-encoder is loaded at MODULE
IMPORT TIME (not lazily on first rerank call) - same eager-loading
pattern as retrieve_direct.py, to keep this cost at app startup instead
of the first user's query.
"""

import math
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from app.config import CROSS_ENCODER_MODEL, RERANK_TOP_K
from app.state import GraphState


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


# ── Eager, module-level construction ────────────────────────
_cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)


def rerank(query: str, chunks: list[Document]) -> list[Document]:
    """
    Scores each chunk against the query, normalizes via sigmoid, returns top-k.
    """
    pairs = [(query, chunk.page_content) for chunk in chunks]
    raw_scores = _cross_encoder.predict(pairs)

    ranked = sorted(zip(chunks, raw_scores), key=lambda pair: pair[1], reverse=True)
    top_chunks = ranked[:RERANK_TOP_K]

    for chunk, raw_score in top_chunks:
        chunk.metadata["rerank_score"] = _sigmoid(float(raw_score))

    return [chunk for chunk, _ in top_chunks]

# Zipping keeps each chunk permanently paired with its score through the sort — 
# there's no risk of the two lists drifting out of alignment,

# key=lambda pair: pair[1]: By default, Python wouldn't know how to sort a tuple that contains a complex Document object. 
# This anonymous lambda function tells Python: "Look at each tuple (pair). 
# Ignore the document at index 0, and base your sorting entirely on the score located at index 1."


def rerank_node(state: GraphState) -> GraphState:
    """
    Idempotent-safe: if chunks were already reranked upstream (e.g. by
    retrieve_multihop, which reranks per sub-query before flattening),
    skip re-scoring - the existing rerank_score is already meaningful
    and re-running rerank() here would score against the wrong query
    (the original compound question, not each chunk's sub-query).
    """
    chunks = state["retrieved_chunks"]

    already_reranked = bool(chunks) and chunks[0].metadata.get("reranked_for_multihop") is True

    if already_reranked:
        print(f"[rerank_node] chunks already reranked upstream, skipping")
        reranked = chunks
    else:
        reranked = rerank(state["query"], chunks)

    return {**state, "reranked_chunks": reranked}


    