# Home Credit Policy Assistant

An agentic RAG (Retrieval-Augmented Generation) pipeline that answers questions about internal company policies and RBI regulatory documents — built for Home Credit India customers and officials who need quick, grounded answers instead of digging through PDFs.

Unlike a simple "embed and retrieve" chatbot, this system classifies each query and routes it through a different retrieval strategy depending on what kind of question it is, reranks candidates with a cross-encoder, gates generation behind a confidence check, and independently verifies every answer against its sources before returning it.

---

## Why not just a standard RAG chain?

A single retrieval strategy doesn't serve every kind of question well:

- *"What's the landline number for the Grievance Redressal Officer?"* — needs exact keyword matching.
- *"Explain HCIN's approach to fair treatment of borrowers"* — needs semantic understanding of a concept, not a keyword.
- *"If a complaint about penal charges isn't resolved, what's the escalation path and what are the fair practice rules on those charges?"* — needs information stitched together from multiple, unrelated parts of the corpus.

This pipeline classifies incoming queries into these three categories and handles each with a purpose-built retrieval path, rather than forcing every question through the same generic similarity search.

---

## Pipeline Architecture

```
                              ┌─────────────────┐
                              │  classify_query  │
                              └────────┬─────────┘
                                       │
                ┌──────────────────────┼──────────────────────┐
                ▼                      ▼                      ▼
        specific_ref            conceptual              multi_hop
                │                      │                      │
                │              ┌───────▼────────┐    ┌────────▼─────────┐
                │              │  retrieve_hyde  │    │ retrieve_multihop│
                │              └───────┬────────┘    │ (decompose, then  │
                │                      │              │  rerank PER      │
                └──────────────────────┤              │  sub-query)       │
                                       ▼              └────────┬─────────┘
                              ┌─────────────────┐              │
                              │ retrieve_direct  │              │
                              │  (BM25 + dense   │              │
                              │   via RRF)       │              │
                              └────────┬─────────┘              │
                                       │                        │
                                       ▼                        │
                              ┌─────────────────┐               │
                              │      rerank      │◄──────────────┘
                              │ (cross-encoder,   │  (idempotent-safe:
                              │  skips if already │   skips if chunks
                              │  reranked)         │   already scored)
                              └────────┬─────────┘
                                       ▼
                              ┌─────────────────┐
                              │ confidence_gate  │
                              └────────┬─────────┘
                          low confidence │ sufficient confidence
                          ┌──────────────┘        └──────────────┐
                          ▼                                      ▼
                  ┌───────────────┐                    ┌─────────────────┐
                  │    abstain     │                    │     generate     │
                  └───────┬───────┘                    └────────┬────────┘
                          │                                     ▼
                          │                          ┌─────────────────────┐
                          │                          │   grounding_check    │
                          │                          │ (independent LLM     │
                          │                          │  fact-check pass)    │
                          │                          └──────────┬──────────┘
                          └──────────────┬──────────────────────┘
                                         ▼
                                ┌─────────────────┐
                                │ format_response  │
                                └─────────────────┘
```

### Query classification
Every query is classified into one of three types before any retrieval happens:
- **`specific_ref`** — names a specific entity (a section, table, contact, defined term).
- **`conceptual`** — asks for an explanation of an idea or process.
- **`multi_hop`** — requires connecting information across multiple sections or documents.

### Three retrieval strategies

**Direct (hybrid) retrieval** combines BM25 (keyword/lexical matching) with dense vector search over a Chroma index, fused with Reciprocal Rank Fusion via `EnsembleRetriever` — so a chunk found by *both* methods is weighted more heavily than one found by only one. This handles `specific_ref` queries, where exact terms matter as much as meaning.

**HyDE retrieval** (Hypothetical Document Embeddings) handles `conceptual` queries. Rather than embedding a short, vague question directly, the pipeline first asks the LLM to write a plausible passage that *would* answer the question, then embeds and searches with that fabricated passage instead. Answer-shaped text sits much closer to real answer chunks in embedding space than a short question does.

**Multi-hop retrieval** decomposes a compound question into 2–4 independent sub-questions, retrieves for each one in parallel, and — critically — **reranks each sub-query's results independently** before merging. Reranking the combined pool in one pass would let one sub-topic's chunks (whichever the compound query's phrasing happens to favor) crowd out another sub-topic entirely, silently dropping half the answer.

### Reranking
All three retrieval paths converge on a cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`), which scores every `(query, chunk)` pair on one consistent scale — something raw BM25/dense scores can't do, since they're produced by different math and aren't directly comparable. Scores are sigmoid-normalized so the confidence threshold stays meaningful regardless of which cross-encoder model is in use.

### Confidence gating
Before generation, the top reranked score is checked against a threshold. If nothing retrieved is confidently relevant, the pipeline **abstains** with a clear message rather than letting the LLM guess — critical for a policy bot, where a wrong answer about compliance or KYC rules is worse than no answer.

### Grounding verification
Every generated answer is independently re-checked by a second LLM pass that verifies each claim is actually supported by the retrieved context — not the same call that generated the answer marking its own homework, but a separate, adversarial "strict fact-checker" pass. Answers that fail this check are still shown, but with an explicit warning attached.

### Explicit path tracking
Rather than inferring how a query was handled from indirect signals (e.g. overloading a `grounded: False` flag to mean two different things), the pipeline records an explicit `path` field (`generated` / `abstained_low_confidence` / `abstained_ungrounded`) in shared state, so the final formatting step never has to guess.

---

## Tech Stack

| Component | Choice |
|---|---|
| Orchestration | LangGraph (`StateGraph`) |
| LLM | Groq (`openai/gpt-oss-120b`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, HuggingFace) |
| Vector store | Chroma (persisted locally) |
| Keyword search | BM25 (`rank_bm25`, persisted via pickle) |
| Retrieval fusion | Reciprocal Rank Fusion (`EnsembleRetriever`) |
| Reranking | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) |
| Evaluation | RAGAS (faithfulness, answer relevancy, context precision, context recall) |
| UI | Streamlit |

---

## Project Structure

```
rag-pipeline/
├── app/
│   ├── config.py              # every tunable constant — models, thresholds, chunk size
│   ├── schemas.py              # Pydantic schemas for structured LLM outputs
│   ├── state.py                 # GraphState — the shared contract every node reads/writes
│   ├── graph.py                  # StateGraph wiring: nodes, edges, routing
│   ├── nodes/
│   │   ├── classify_query.py      # query classification + routing
│   │   ├── retrieve_direct.py      # hybrid BM25 + dense retrieval (RRF)
│   │   ├── retrieve_hyde.py         # hypothetical document generation
│   │   ├── retrieve_multihop.py      # query decomposition + parallel retrieval
│   │   ├── rerank.py                  # cross-encoder reranking (idempotent-safe)
│   │   ├── confidence_gate.py          # routing + abstain logic
│   │   ├── generate.py                  # grounded answer generation
│   │   ├── grounding_check.py            # independent faithfulness verification
│   │   └── format_response.py             # final response shaping
│   └── ingestion/
│       ├── loader.py                       # PDF loading
│       ├── chunker.py                       # text splitting
│       └── build_vectorstore.py              # embeds chunks, builds Chroma + BM25 indexes
├── evals/
│   ├── golden_eval_set.py     # hand-written Q&A pairs spanning all query types
│   └── ragas_eval.py          # runs the pipeline against the golden set, scores with RAGAS
├── tests/                    # isolated tests for every node, run independently of the graph
├── ui/
│   └── streamlit_app.py       # chat interface
├── data/
│   ├── raw_pdfs/                # source policy/regulatory documents
│   ├── chroma_db/                 # persisted vector index
│   └── bm25_index.pkl               # persisted keyword index
└── main.py                    # compiles the graph, exposes ask(query) -> str
```

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add GROQ_API_KEY
```

Add source PDFs to `data/raw_pdfs/`, then build the indexes:

```bash
python -m app.ingestion.build_vectorstore
```

Run a query from the command line:

```bash
python main.py
```

Or launch the chat UI:

```bash
streamlit run ui/streamlit_app.py
```

---

## Testing

Every node is tested in isolation before being wired into the graph — this is how two real bugs were caught during development:

1. **A Python closure trap** in the multi-hop fan-out, where every parallel retrieval branch was silently searching for the same (last) sub-query instead of its own, due to late variable binding in a loop.
2. **A cross-encoder reranking bug** in the multi-hop path, where reranking all sub-queries' candidates together in one pass let one sub-topic dominate the results and silently drop the other — fixed by reranking per sub-query before merging.

Run any node's isolated test independently:

```bash
python -m tests.test_classify_query
python -m tests.test_retrieve_multihop
python -m tests.test_rerank
python -m tests.test_confidence_gate
python -m tests.test_generate
python -m tests.test_grounding_check
```

---

## Evaluation

`evals/golden_eval_set.py` contains hand-verified question/answer pairs spanning `specific_ref`, `conceptual`, and `multi_hop` query types across the ingested corpus. `evals/ragas_eval.py` runs the full pipeline against this set and scores results on:

- **Faithfulness** — do the answer's claims trace back to retrieved context?
- **Answer relevancy** — does the answer address the question actually asked?
- **Context precision** — how much of what was retrieved was actually useful?
- **Context recall** — did retrieval find everything needed to fully answer?

```bash
python -m evals.ragas_eval
```
