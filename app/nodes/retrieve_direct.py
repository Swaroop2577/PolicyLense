"""
retrieve_direct.py

Handles retrieval for specific_ref queries - the "user knows exactly what
they want" case (e.g. "what does Section 4.2 say about X"). Combines two
retrieval methods: BM25 (keyword/lexical match) and dense (semantic/vector
match) via RRF, then merges their results.

Both retrievers are built at MODULE IMPORT TIME (not lazily on first
query) - this eliminates any first-query loading delay, since import
happens once when app/graph.py builds the graph at startup.
"""

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from app.config import TOP_K_PER_RETRIEVER
from app.state import GraphState
from app.ingestion.build_vectorstore import load_vectorstore, load_bm25_index


# ── Eager, module-level construction ────────────────────────
# Runs once, the moment this module is first imported.

_bm25_retriever = load_bm25_index()
_bm25_retriever.k = TOP_K_PER_RETRIEVER

_dense_retriever = load_vectorstore().as_retriever(
    search_kwargs={"k": TOP_K_PER_RETRIEVER}
)

_ensemble_retriever = EnsembleRetriever(
    retrievers=[_bm25_retriever, _dense_retriever],
    weights=[0.5, 0.5],
)


def retrieve_direct(query: str) -> list[Document]:
    """
    Runs hybrid retrieval via RRF. No manual dedup - EnsembleRetriever's
    internal deduplication (by page_content hash) is what produces the
    correct fused rank; re-deduplicating after would only mask a broken merge.
    """
    return _ensemble_retriever.invoke(query)


def retrieve_direct_node(state: GraphState) -> GraphState:
    """
    LangGraph node. Uses hyde_passage for retrieval if it exists (conceptual
    path), otherwise the raw query (specific_ref path). Always writes back
    to retrieved_chunks - the original query is preserved separately for
    generation later.
    """
    retrieval_query = state.get("hyde_passage") or state["query"]
    chunks = retrieve_direct(retrieval_query)
    return {**state, "retrieved_chunks": chunks}