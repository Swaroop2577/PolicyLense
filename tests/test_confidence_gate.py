"""
Run with:
    python -m tests.test_confidence_gate

Tests both branches: a query that SHOULD pass the gate (relevant to your
PDFs) and one that SHOULD fail it (nonsense/off-topic), to confirm
routing actually branches both ways, not just one.
"""

from app.nodes.retrieve_direct import retrieve_direct
from app.nodes.rerank import rerank
from app.nodes.confidence_gate import route_by_confidence, abstain_node
from app.state import GraphState

TEST_CASES = [
    ("What is the process for filing a customer grievance?", "expected: generate"),
    ("What is the capital of France and how many moons does Jupiter have?", "expected: abstain"),
]

if __name__ == "__main__":
    for query, expectation in TEST_CASES:
        candidates = retrieve_direct(query)
        reranked = rerank(query, candidates)

        state: GraphState = {"query": query, "reranked_chunks": reranked}
        decision = route_by_confidence(state)

        top_score = reranked[0].metadata.get("rerank_score") if reranked else None

        print(f"Query: {query}")
        print(f"  top_score: {top_score}")
        print(f"  routed to: {decision}  ({expectation})")

        if decision == "abstain":
            result_state = abstain_node(state)
            print(f"  abstain message: {result_state['answer']}")
        print()