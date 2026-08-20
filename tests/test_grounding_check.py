"""
Run with:
    python -m tests.test_grounding_check

Tests two cases:
1. A real generated answer (should come back grounded=True)
2. A deliberately fabricated answer with a made-up claim (should come
   back grounded=False) - this proves the checker can actually catch
   a bad answer, not just rubber-stamp everything as grounded.
"""

from app.nodes.retrieve_direct import retrieve_direct
from app.nodes.rerank import rerank
from app.nodes.generate import generate_answer
from app.nodes.grounding_check import check_grounding

TEST_QUERY = "What is the process for filing a customer grievance?"

if __name__ == "__main__":
    candidates = retrieve_direct(TEST_QUERY)
    reranked = rerank(TEST_QUERY, candidates)

    # Case 1: real generated answer
    real_answer = generate_answer(TEST_QUERY, reranked)
    result_real = check_grounding(real_answer, reranked)

    print("=== Case 1: Real generated answer ===")
    print(f"Answer: {real_answer}\n")
    print(f"Grounded: {result_real.grounded}")
    print(f"Unsupported claims: {result_real.unsupported_claims}\n")

    # Case 2: deliberately fabricated answer
    fake_answer = (
        "Customer grievances must be filed within 24 hours via the company's "
        "dedicated WhatsApp hotline, and Home Credit guarantees resolution "
        "within 2 hours or the customer receives automatic compensation of Rs. 5000."
    )
    result_fake = check_grounding(fake_answer, reranked)

    print("=== Case 2: Fabricated answer (should be flagged) ===")
    print(f"Answer: {fake_answer}\n")
    print(f"Grounded: {result_fake.grounded}")
    print(f"Unsupported claims: {result_fake.unsupported_claims}")