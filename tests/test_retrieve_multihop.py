"""
Run with:
    python -m tests.test_retrieve_multihop
"""

from app.nodes.retrieve_multihop import retrieve_multihop, _decompose_chain

# A genuinely multi-hop question for your policy docs - connects two
# different sections/topics that likely live in different parts of the PDFs
TEST_QUERY = (
    "If a customer's complaint about penal charges isn't resolved by Home "
    "Credit, what's the full escalation path, and what are HCIN's fair "
    "practice commitments regarding those charges in the first place?"
)

if __name__ == "__main__":
    # First, see exactly what sub-queries decomposition produced
    sub_queries = _decompose_chain.invoke({"query": TEST_QUERY}).sub_queries
    print("--- Decomposed sub-queries ---")
    for sq in sub_queries:
        print(f"  - {sq}")
    print()

    chunks = retrieve_multihop(TEST_QUERY)

    print(f"Query: {TEST_QUERY}")
    print(f"Total chunks retrieved: {len(chunks)}\n")

    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i} (source: {chunk.metadata.get('source_file')}, "
              f"page: {chunk.metadata.get('page')}) ---")
        print(chunk.page_content[:200])
        print()