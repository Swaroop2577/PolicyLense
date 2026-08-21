"""
generate.py

Produces the final answer from the top reranked chunks. This is where
the LLM finally sees retrieved context and writes a response - every
prior node existed to get good chunks to this point.
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from app.config import GROQ_API_KEY, GENERATION_MODEL, GENERATION_TEMPERATURE
from app.state import GraphState


def _format_context(chunks: list[Document]) -> str:
    """
    Turns a list of chunks into a single numbered context block, so the
    LLM can cite which chunk supports which part of its answer.
    """
    return "\n\n".join(
        f"[{i+1}] (Source: {chunk.metadata.get('source_file')}, "
        f"Page: {chunk.metadata.get('page')})\n{chunk.page_content}"
        for i, chunk in enumerate(chunks)
    )



_llm = ChatGroq(
    model=GENERATION_MODEL,
    temperature=GENERATION_TEMPERATURE,
    api_key=GROQ_API_KEY,
)

_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a policy assistant answering questions using ONLY the
provided context. Follow these rules strictly:

1. Base your answer ONLY on the context below - do not use outside knowledge.
2. If the context does not contain enough information to answer, say so clearly
   instead of guessing.
3. Cite the source number(s) you used in this EXACT format: [1] or [1][3].
   Do NOT use any other citation format. Specifically, NEVER use bracket-dagger
   style citations like 【1†L1-L5】 - only plain square brackets with numbers.
4. Be concise and direct - this is a policy reference tool, not a conversation.

Context:
{context}"""),
    ("human", "{query}"),
])

_generation_chain = _prompt | _llm | StrOutputParser()



def generate_answer(query: str, chunks: list[Document]) -> str:
    """
    Generates a grounded answer from the query and its reranked chunks.
    """
    context = _format_context(chunks)
    return _generation_chain.invoke({"query": query, "context": context})


def generate_node(state: GraphState) -> GraphState:
    answer = generate_answer(state["query"], state["reranked_chunks"])
    return {**state, "answer": answer, "grounded": None}

"""
Why "grounded": None here
Setting it to None here (rather than defaulting to True) is a deliberate signal: "not yet verified," so nothing downstream 
could mistake an unverified answer for a confirmed one if the graph wiring ever skipped the grounding step by mistake.
"""

