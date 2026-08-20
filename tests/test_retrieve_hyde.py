"""
Test retrieve_hyde.py in isolation - no LangGraph needed yet.

Run with:
    python -m tests.test_retrieve_hyde
"""

from app.nodes.retrieve_hyde import generate_hyde_passage

TEST_QUERIES = [
    "Explain how attention mechanisms work in transformers",
    "What is gradient boosting and why is it used for tabular data?",
]

if __name__ == "__main__":
    for q in TEST_QUERIES:
        passage = generate_hyde_passage(q)
        print(f"\nQuery : {q}")
        print(f"Passage:\n{passage}")