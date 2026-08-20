from langchain_community.document_loaders import PyMuPDFLoader
import os

# Point this directly to your Fair Practice Code PDF
pdf_path = os.path.join("data", "raw_pdfs", "Fair-Practice-Code-English.pdf")

# Use PyMuPDFLoader instead of PyPDFLoader
loader = PyMuPDFLoader(pdf_path)
pages = loader.load()

# Print the raw text of Page 2 (index 1)
print("--- RAW TEXT OF PAGE 2 ---")
print(pages[1].page_content)