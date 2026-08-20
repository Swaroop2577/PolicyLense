"""
chunker.py

Splits loaded PDF pages into smaller overlapping chunks. This is the
second stage of ingestion - runs right after loader.py, before embedding.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Splits a list of page-level Documents into smaller chunk-level Documents.
    Each chunk keeps the metadata (source_file, page) of the page it came from,
    plus a new chunk_id for traceability.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    print(f"[chunker] {len(documents)} pages -> {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    return chunks


if __name__ == "__main__":
    from app.ingestion.loader import load_pdfs

    pages = load_pdfs()
    chunks = chunk_documents(pages)

    print(f"\nFirst chunk preview:\n{chunks[0].page_content[:300]}")
    print(f"\nMetadata: {chunks[0].metadata}")