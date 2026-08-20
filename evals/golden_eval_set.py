"""
golden_eval_set.py

Hand-written test cases spanning all 3 query types and multiple source
PDFs. Each entry has:
  - query: the question to ask the pipeline
  - query_type: expected classification (sanity-checks classify_query too)
  - ground_truth: a human-written correct answer (RAGAS compares
    generated answers against this for faithfulness/relevance scoring)
  - expected_sources: which PDF(s) the answer should draw from - lets us
    check retrieval quality, not just final answer quality

RULE: every query here must be answerable from the current corpus. If
you add/remove PDFs, review this file - a golden question about content
that no longer exists will make the eval unfairly fail.
"""

from dataclasses import dataclass, field


@dataclass
class GoldenExample:
    query: str
    query_type: str  # "specific_ref" | "conceptual" | "multi_hop"
    ground_truth: str
    expected_sources: list[str] = field(default_factory=list)



GOLDEN_EXAMPLES: list[GoldenExample] = [

    # ── specific_ref: names a specific channel/contact/section ──────
    GoldenExample(
        query="What is the landline number for the Grievance Redressal Officer?",
        query_type="specific_ref",
        ground_truth=(
            "The Grievance Redressal Officer can be reached at landline "
            "0124 6628713."
        ),
        expected_sources=["Grievance_Redressal_Policy_2020.pdf", "Fair-Practice-Code-English.pdf"],
    ),

    GoldenExample(
        query="Within how many days must HCIN release original movable property documents after loan closure?",
        query_type="specific_ref",
        ground_truth=(
            "HCIN must release all original movable property documents "
            "within 30 days after full repayment or settlement of the loan account."
        ),
        expected_sources=["Fair-Practice-Code-English.pdf"],  # or the dedicated property-release circular, if ingested separately
    ),

    # ── conceptual: explain an idea, not a single fact ──────────────
    GoldenExample(
        query="What does the Reserve Bank - Integrated Ombudsman Scheme provide for customers?",
        query_type="conceptual",
        ground_truth=(
            "The RB-IOS gives customers a free, cost-effective escalation "
            "path to the RBI if their complaint isn't resolved by the "
            "lender within a specified period, with a Nodal Officer "
            "representing the company before the Ombudsman."
        ),
        expected_sources=["RB-IOS_2021.pdf"],
    ),

    GoldenExample(
        query="What ethical standards must recovery agents follow when collecting overdue loans?",
        query_type="conceptual",
        ground_truth=(
            "Recovery agents must avoid intimidation, harassment, public "
            "humiliation, or contacting borrowers before 8am or after 7pm, "
            "and must not make false or misleading representations."
        ),
        expected_sources=["Fair-Practice-Code-English.pdf"],
    ),

    # ── multi_hop: genuinely needs 2+ documents/sections ─────────────
    GoldenExample(
        query=(
            "If a customer's complaint about a recovery agent's conduct "
            "isn't resolved by Home Credit, what's the escalation path, "
            "and what specific ethical rules govern recovery agents in "
            "the first place?"
        ),
        query_type="multi_hop",
        ground_truth=(
            "Escalation: unresolved complaints can be appealed to the "
            "RBI's NBFC Ombudsman or under the RB-IOS. Recovery agent "
            "conduct rules: no intimidation, harassment, or contact "
            "outside 8am-7pm, and outsourcing agents must follow the "
            "same code of conduct as employees, per RBI's outsourcing "
            "guidelines."
        ),
        expected_sources=[
            "Grievance_Redressal_Policy_2020.pdf",
            "Fair-Practice-Code-English.pdf",
            "Outsourcing_Recovery_Agents.pdf",
        ],
    ),

    GoldenExample(
        query=(
            "How does the Key Facts Statement relate to the digital "
            "lending cooling-off period requirements?"
        ),
        query_type="multi_hop",
        ground_truth=(
            "The KFS must disclose the recovery mechanism, grievance "
            "officer, and cooling-off period before loan execution; the "
            "Digital Lending Directions separately mandate a minimum "
            "3-day cooling-off period during which the borrower can exit "
            "the loan by repaying principal plus proportionate APR, "
            "without penalty."
        ),
        expected_sources=["Fair-Practice-Code-English.pdf", "Digital_Lending_Directions_2025.pdf"],
    ),
]