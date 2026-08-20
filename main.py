"""
main.py

Entry point. Compiles the graph and runs a query through it end-to-end.
"""

from app.graph import build_graph

app = build_graph()


def ask(query: str) -> str:
    result = app.invoke({"query": query})
    return result["answer"]


if __name__ == "__main__":
    test_queries = [
        "What is the process for filing a customer grievance?",              # specific_ref
        "Explain HCIN's approach to fair treatment of borrowers",             # conceptual
        "If a customer's complaint about penal charges isn't resolved by "
        "Home Credit, what's the full escalation path, and what are HCIN's "
        "fair practice commitments regarding those charges in the first "
        "place?",                                                            # multi_hop
    ]

    for q in test_queries:
        print(f"\n{'='*60}\nQuery: {q}\n{'='*60}")
        print(ask(q))