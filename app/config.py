"""
Every tunable constant in the pipeline lives here. When you want to change
a model name, a threshold, or chunk size, you should only ever need to
edit this one file — never hunt through node files.

Fields are grouped by the section that introduces them. Right now only
the classification section's constants are filled in; later sections
will add their own group below.
"""

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

CLASSIFIER_MODEL = "openai/gpt-oss-120b"
CLASSIFIER_TEMPERATURE = 0

HYDE_MODEL = "openai/gpt-oss-120b"
HYDE_TEMPERATURE = 0.3

# ── Ingestion (loader.py, chunker.py, build_vectorstore.py) ────
RAW_PDFS_DIR = "data/raw_pdfs"
CHROMA_DIR = "data/chroma_db"
 
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Retrieval (added in the retrieval section) ─────────────────
TOP_K_PER_RETRIEVER = 10

# ── Reranking (added later) ─────────────────────────────────────
RERANK_TOP_K = 5
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Confidence gate (added later) ───────────────────────────────
CONFIDENCE_THRESHOLD = 0.05

# ── Generation (added later) ────────────────────────────────────
GENERATION_MODEL = "openai/gpt-oss-120b"
GENERATION_TEMPERATURE = 0.1


BM25_INDEX_PATH = "data/bm25_index.pkl"
