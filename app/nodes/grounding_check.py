"""
grounding_check.py

Verifies that the generated answer's claims are actually supported by the
retrieved chunks, not hallucinated. Uses the LLM itself as a judge - a
technique often called "LLM-as-judge" or NLI-style entailment checking.
"""

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from app.config import GROQ_API_KEY, CLASSIFIER_MODEL, CLASSIFIER_TEMPERATURE
from app.state import GraphState
from app.nodes.generate import _format_context



class GroundingResult(BaseModel):
    grounded: bool = Field(
        description="True if every claim in the answer is supported by the context"
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Specific claims in the answer NOT found in the context, if any"
    )

"""
Why ask for unsupported_claims, not just a bare True/False? A bare boolean tells you that something's wrong but not what. 
Capturing the specific unsupported claim is what makes this debuggable later

default_factory=list (rather than a required field) means when grounded=True, this can just be an empty list

"""

_llm = ChatGroq(
    model=CLASSIFIER_MODEL,
    temperature=CLASSIFIER_TEMPERATURE,
    api_key=GROQ_API_KEY,
).with_structured_output(GroundingResult)

_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a strict fact-checker. Given a context and an answer
that claims to be based on that context, determine whether EVERY factual claim
in the answer is actually supported by the context. Do not judge whether the
answer is well-written - only whether it's faithful to the context.

Context:
{context}

Answer to check:
{answer}"""),
    ("human", "Is this answer fully grounded in the context?"),
])

_grounding_chain = _prompt | _llm



def check_grounding(answer: str, chunks: list[Document]) -> GroundingResult:
    context = _format_context(chunks)
    return _grounding_chain.invoke({"context": context, "answer": answer})



def grounding_check_node(state: GraphState) -> GraphState:
    result = check_grounding(state["answer"], state["reranked_chunks"])

    if not result.grounded:
        print(f"[grounding_check] unsupported claims: {result.unsupported_claims}")

    return {
        **state,
        "grounded": result.grounded,
        "path": "generated" if result.grounded else "abstained_ungrounded",
    }