"""
ragas_eval.py

Runs the compiled graph against golden_eval_set.py, then scores each
result with RAGAS metrics. Uses app.graph directly (not main.ask()) -
RAGAS needs the RETRIEVED CONTEXTS alongside the answer, and ask()
only returns the final formatted string, discarding everything else.
"""

import sys
import types

# Hack to bypass ragas crash on modern langchain-community versions
dummy_chat = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_chat.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy_chat


from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from app.config import GROQ_API_KEY, CLASSIFIER_MODEL, EMBEDDING_MODEL
from ragas.run_config import RunConfig

import json
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

from app.graph import build_graph
from app.config import GROQ_API_KEY, CLASSIFIER_MODEL
from evals.golden_eval_set import GOLDEN_EXAMPLES

# ── RAGAS judge LLM + embeddings, wrapping your existing Groq/HF setup ──
_ragas_llm = LangchainLLMWrapper(
    ChatGroq(model=CLASSIFIER_MODEL, temperature=0, api_key=GROQ_API_KEY)
)

_ragas_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
)

METRICS = [
    Faithfulness(),
    AnswerRelevancy(strictness=1),   # strictness=1 means "generate 1 variation, not 3" - avoids Groq's n>1 rejection
    ContextPrecision(),
    ContextRecall(),
]



def _run_pipeline_on_golden_set(app) -> list[dict]:
    """
    Runs every golden example through the compiled graph, collecting
    exactly the fields RAGAS's evaluate() expects: question, answer,
    contexts (list of strings), and ground_truth.
    """
    records = []

    for example in GOLDEN_EXAMPLES:
        result = app.invoke({"query": example.query})

        contexts = [chunk.page_content for chunk in result.get("reranked_chunks", [])]

        records.append({
            "question": example.query,
            "answer": result["answer"],
            "contexts": contexts,
            "ground_truth": example.ground_truth,
        })

        print(f"[ragas_eval] ran: {example.query[:60]}... "
              f"(type={example.query_type}, {len(contexts)} contexts)")

    return records



def run_eval() -> None:
    app = build_graph()

    records = _run_pipeline_on_golden_set(app)
    dataset = Dataset.from_list(records)

    print("[ragas_eval] scoring with RAGAS...")
    results = evaluate(
        dataset,
        metrics=METRICS,
        llm=_ragas_llm,
        embeddings=_ragas_embeddings,
        run_config=RunConfig(timeout=120, max_workers=2),
    )

    df = results.to_pandas()
    print("\n--- RAGAS Results ---")
    # print(df[["question", "faithfulness", "answer_relevancy",
    #           "context_precision", "context_recall"]].to_string(index=False))
    print(df.columns.tolist())  # run once to confirm actual column names
    print(df.to_string(index=False))
    _save_results(df)


def _save_results(df) -> None:
    results_dir = Path("evals/results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"ragas_run_{timestamp}.json"

    df.to_json(output_path, orient="records", indent=2)
    print(f"\n[ragas_eval] results saved to {output_path}")


if __name__ == "__main__":
    run_eval()