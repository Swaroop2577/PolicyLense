"""
GraphState is the single shared contract every node reads from and writes to.
LangGraph merges each node's returned dict into this state as the query
flows through the graph.

RULE: every new section ADDS fields here. Never rename or remove a field
once another node depends on it — treat this file like an API contract
that the whole pipeline agrees on.
"""

# 1. Optional expects exactly ONE type
# The Optional keyword is designed to take a single data type and tell Python, "This can be this type, OR it can be None."
# Because you passed it three different strings separated by commas, Python gets confused. It expects something like Optional[str] or Optional[int].

# 2. Optional expects TYPES, not VALUES
# Literal is special: it is the only type hint that accepts raw values (like specific strings or numbers).


from typing import Literal, Optional, TypedDict


class GraphState(TypedDict):
    # ── classify_query.py ───────────────────────────────────
    query: str
    query_type: Optional[Literal["specific_ref", "conceptual", "multi_hop"]]

    # ── retrieve_hyde.py (added next) ───────────────────────
    hyde_passage: Optional[str]

    # ── retrieve_direct.py / retrieve_multihop.py (added later) ──
    retrieved_chunks: Optional[list]

    # ── rerank.py (added later) ─────────────────────────────
    reranked_chunks: Optional[list]

    # ── generate.py / grounding_check.py (added later) ──────
    answer: Optional[str]
    grounded: Optional[bool]

     # ── confidence_gate.py / grounding_check.py ────────────────
    # explicitly records WHICH terminal path the query took, so
    # format_response.py never has to guess from message content
    path: Optional[Literal["generated", "abstained_low_confidence", "abstained_ungrounded"]]
    