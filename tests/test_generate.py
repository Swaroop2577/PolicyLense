"""
Run with:
    python -m tests.test_generate

Tests generate_answer() using real reranked chunks, so we're testing
against actual retrieval quality, not fake context.
"""

from app.nodes.retrieve_direct import retrieve_direct
from app.nodes.rerank import rerank
from app.nodes.generate import generate_answer

TEST_QUERY = "What is the process for filing a customer grievance?"

if __name__ == "__main__":
    candidates = retrieve_direct(TEST_QUERY)
    reranked = rerank(TEST_QUERY, candidates)

    print(f"Query: {TEST_QUERY}")
    print(f"Using {len(reranked)} reranked chunks as context\n")

    answer = generate_answer(TEST_QUERY, reranked)

    print("--- Generated Answer ---")
    print(answer)