"""
graph.py

Wires every node and edge built across all prior sections into one
compiled LangGraph StateGraph. This is the only file that knows the
full shape of the pipeline - every node file above stays ignorant of
its neighbors.
"""

from langgraph.graph import StateGraph, END

from app.state import GraphState
from app.nodes.classify_query import classify_query_node
from app.nodes.retrieve_direct import retrieve_direct_node
from app.nodes.retrieve_hyde import retrieve_hyde_node
from app.nodes.retrieve_multihop import retrieve_multihop_node
from app.nodes.rerank import rerank_node
from app.nodes.confidence_gate import route_by_confidence, abstain_node
from app.nodes.generate import generate_node
from app.nodes.grounding_check import grounding_check_node
from app.nodes.format_response import format_response_node
from app.nodes.classify_query import classify_query_node, route_by_query_type



def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("classify_query", classify_query_node)
    graph.add_node("retrieve_direct", retrieve_direct_node)
    graph.add_node("retrieve_hyde", retrieve_hyde_node)
    graph.add_node("retrieve_multihop", retrieve_multihop_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("abstain", abstain_node)
    graph.add_node("generate", generate_node)
    graph.add_node("grounding_check", grounding_check_node)
    graph.add_node("format_response", format_response_node)


    graph.set_entry_point("classify_query")

    # classify_query branches 3 ways based on query_type
    graph.add_conditional_edges(
        "classify_query",
        route_by_query_type,
        {
            "retrieve_direct": "retrieve_direct",
            "retrieve_hyde": "retrieve_hyde",
            "retrieve_multihop": "retrieve_multihop",
        },
    )

    # hyde generates a passage, THEN still needs to retrieve with it
    graph.add_edge("retrieve_hyde", "retrieve_direct")

    # both direct and multihop retrieval converge on rerank
    graph.add_edge("retrieve_direct", "rerank")
    graph.add_edge("retrieve_multihop", "rerank")

    # rerank branches based on top score
    graph.add_conditional_edges(
        "rerank",
        route_by_confidence,
        {
            "generate": "generate",
            "abstain": "abstain",
        },
    )

    # generate always gets fact-checked
    graph.add_edge("generate", "grounding_check")

    # both success and failure paths converge on formatting
    graph.add_edge("grounding_check", "format_response")
    graph.add_edge("abstain", "format_response")

    graph.add_edge("format_response", END)

    return graph.compile()