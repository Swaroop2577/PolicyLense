"""
confidence_gate.py

Routing function (not a content-producing node) that inspects the top
reranked chunk's score and decides: proceed to generation, or abstain.
Prevents the bot from confidently answering when nothing relevant was
actually found - critical for a policy bot where a wrong answer about
compliance/KYC rules is worse than no answer.
"""

from app.config import CONFIDENCE_THRESHOLD
from app.state import GraphState



def route_by_confidence(state: GraphState) -> str:
    """
    LangGraph conditional-edge function. Returns the name of the next
    node based on the top reranked chunk's score.
    """
    if not state["reranked_chunks"]:       #If retrieval genuinely found nothing (empty list), indexing [0] would crash
        return "abstain"

    top_score = state["reranked_chunks"][0].metadata.get("rerank_score", 0)
    return "generate" if top_score >= CONFIDENCE_THRESHOLD else "abstain"


# Notice route_by_confidence doesn't return GraphState and doesn't do {**state, ...} — it returns a plain string 
# ("generate" or "abstain"). This is a conditional edge function,


def abstain_node(state: GraphState) -> GraphState:
    """
    Terminal node when confidence is too low to generate. Produces a
    clear refusal instead of letting the LLM guess.
    """
    return {
        **state,
        "answer": (
            "I couldn't find a confident answer to this in the available "
            "policy documents. You may want to rephrase your question or "
            "check with an official Home Credit representative."
        ),
        "grounded": False,
        "path": "abstained_low_confidence",
    }