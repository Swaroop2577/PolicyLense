"""
build_vectorstore.py

Final ingestion stage. Takes chunked Documents, embeds them with Gemini's
embedding model, and persists them into a local Chroma index on disk.

This is a ONE-OFF SCRIPT - you run it once (or whenever your PDFs change),
not on every query. The retrieval nodes (built later) just open the
already-built index from disk.
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import EMBEDDING_MODEL, CHROMA_DIR
from app.ingestion.loader import load_pdfs
from app.ingestion.chunker import chunk_documents
from dotenv import load_dotenv

import pickle
from langchain_community.retrievers import BM25Retriever
from app.config import BM25_INDEX_PATH


load_dotenv()


def build_vectorstore() -> Chroma:
    """
    Runs the full ingestion pipeline: load -> chunk -> embed -> persist.
    Returns the Chroma vectorstore instance (also saved to disk at CHROMA_DIR).
    """
    pages = load_pdfs()
    chunks = chunk_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"[build_vectorstore] embedding {len(chunks)} chunks into Chroma...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    print(f"[build_vectorstore] done. Index saved to {CHROMA_DIR}")
    return vectorstore


def load_vectorstore() -> Chroma:
    """
    Opens an already-built Chroma index from disk WITHOUT re-embedding.
    This is what retrieval nodes (retrieve_direct.py etc.) will call later -
    they never call build_vectorstore() directly.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    return Chroma(                           
        persist_directory=CHROMA_DIR,         #This is the important design decision in this file: two separate functions, 
        embedding_function=embeddings,        #not one. build_vectorstore() re-embeds everything from scratch 
        )                                      # (expensive, run rarely). load_vectorstore() just opens the index 
                                              # that's already on disk (cheap, instant).




def build_bm25_index() -> BM25Retriever:
    """
    Builds a BM25 retriever from the same chunks used for Chroma, and
    pickles it to disk. Run once during ingestion, not per-query -
    BM25Retriever has no native persistence, so this is the workaround.
    """
    pages = load_pdfs()
    chunks = chunk_documents(pages)

    bm25 = BM25Retriever.from_documents(chunks)

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)

    print(f"[build_bm25_index] BM25 index saved to {BM25_INDEX_PATH}")
    return bm25


def load_bm25_index() -> BM25Retriever:
    """
    Loads the pickled BM25 retriever from disk - instant, no PDF
    re-loading or re-chunking involved.
    """
    with open(BM25_INDEX_PATH, "rb") as f:
        return pickle.load(f)       


if __name__ == "__main__":
    vectorstore = build_vectorstore()
    build_bm25_index()

    # quick sanity check - run one similarity search right after building
    results = vectorstore.similarity_search("What is the main topic of this document?", k=2)
    print(f"\nSanity check - top result:\n{results[0].page_content[:300]}")
    print(f"\nMetadata: {results[0].metadata}")