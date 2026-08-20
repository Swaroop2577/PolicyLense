"""
Test classify_query.py in isolation - no LangGraph needed yet.

Run with:
    python -m tests.test_classify_query
"""

from app.nodes.classify_query import classify_query

TEST_QUERIES = [
    "What does Table 3 show about recall scores?",              # expect: specific_ref
    "Explain how attention mechanisms work in transformers",     # expect: conceptual
    "How does the method in Section 2 relate to the results in Section 4?",  # expect: multi_hop
]

if __name__ == "__main__":
    for q in TEST_QUERIES:
        result = classify_query(q)
        print(f"\nQuery : {q}")
        print(f"Type  : {result.query_type}")
        print(f"Reason: {result.reasoning}")