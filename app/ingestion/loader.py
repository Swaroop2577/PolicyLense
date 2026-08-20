"""
loader.py

Loads all PDFs from data/raw_pdfs/ into LangChain Document objects.
This is the first stage of ingestion - runs once, offline, before the
pipeline ever answers a query.
"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from app.config import RAW_PDFS_DIR


def load_pdfs(pdf_dir: str = RAW_PDFS_DIR) -> list[Document]:
    """
    Loads every .pdf file in pdf_dir, returns a flat list of Documents
    (one Document per page, across all PDFs).
    """
    pdf_dir_path = Path(pdf_dir)
    pdf_files = sorted(pdf_dir_path.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {pdf_dir_path.resolve()}. "
            f"Add some .pdf files there before running ingestion."
        )

    all_documents: list[Document] = []    #This initializes an empty master list called all_documents. 
                                          #The type hint list[Document] indicates it will hold Document objects 
                                          # (a standard format in LangChain that stores text and its associated metadata).

    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        for page in pages:
            page.metadata["source_file"] = pdf_path.name

        all_documents.extend(pages)  #If you had used append(), Python would take the entire pages list and shove it 
                                    #into all_documents as a single item. You would end up with a nested list 
                                    # (a list inside a list), which breaks most Langchain tools:
                                    #Bad (Append): [ [Doc1, Doc2, Doc3...] ]

                                    #By using extend(), you are telling Python to unpack the pages list and add each Document 
                                    # individually to the end of all_documents. It keeps everything flat.
                                    #Good (Extend): [Doc1, Doc2, Doc3...]                       

        print(f"[loader] loaded {len(pages)} pages from {pdf_path.name}")

    print(f"[loader] total: {len(all_documents)} pages from {len(pdf_files)} PDFs")
    return all_documents


if __name__ == "__main__":
    docs = load_pdfs()
    print(f"\nFirst document preview:\n{docs[0].page_content[:300]}")
    print(f"\nMetadata: {docs[0].metadata}")