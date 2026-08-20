"""
Run with:
    python -m tests.test_rerank
"""

from app.nodes.retrieve_direct import retrieve_direct
from app.nodes.rerank import rerank

TEST_QUERY = "What is the process for filing a customer grievance?"

if __name__ == "__main__":
    candidates = retrieve_direct(TEST_QUERY)
    print(f"Retrieved {len(candidates)} candidates before reranking\n")

    reranked = rerank(TEST_QUERY, candidates)
    print(f"Reranked down to top {len(reranked)}\n")

    for i, chunk in enumerate(reranked):
        score = chunk.metadata.get("rerank_score")
        print(f"--- Rank {i} | score: {score:.4f} "
              f"(source: {chunk.metadata.get('source_file')}) ---")
        print(chunk.page_content[:500])
        print()