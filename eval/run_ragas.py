import os
import json
from typing import List, Dict, Any, Callable

from dotenv import load_dotenv
import pandas as pd
from langchain_groq import ChatGroq

from src.vectorstore import FaissVectorStore
from src.hybrid_search import HybridRetriever
from src.reranker import CrossEncoderReranker
from src.compressor import ContextualCompressor
from eval.custom_metrics import evaluate_row

load_dotenv()

LLM_MODEL = "llama-3.1-8b-instant"


def load_testset(path: str = "eval/qa_testset.json") -> List[Dict[str, str]]:
    with open(path, "r") as f:
        return json.load(f)


def generate_answer(llm: ChatGroq, question: str, contexts: List[str]) -> str:
    context_block = "\n\n".join(contexts) if contexts else "No context retrieved."
    prompt = (
        f"Answer the question using the context below. The context may not "
        f"use the exact same words as the question - use reasonable "
        f"inference from what IS stated to answer if the underlying "
        f"information is present. Only say you don't know if the context "
        f"truly contains nothing relevant to the question.\n\n"
        f"Context:\n{context_block}\n\nQuestion: {question}\nAnswer:"
    )
    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content.strip()


def run_baseline_pipeline(
    store: FaissVectorStore, llm: ChatGroq, question: str, top_k: int = 5
) -> Dict[str, Any]:
    """Plain FAISS similarity search -> generate. No hybrid, no rerank, no compression."""
    results = store.query(question, top_k=top_k)
    contexts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
    answer = generate_answer(llm, question, contexts)
    return {"answer": answer, "contexts": contexts}


def run_advanced_pipeline(
    hybrid: HybridRetriever,
    reranker: CrossEncoderReranker,
    compressor: ContextualCompressor,
    llm: ChatGroq,
    question: str,
    candidate_n: int = 40,
    rerank_k: int = 15,
    final_k: int = 6,
    min_contexts_after_compression: int = 3,
) -> Dict[str, Any]:
    """
    Full pipeline: hybrid search -> rerank -> contextual compression -> generate.

    Safety net: contextual compression can occasionally over-trim and drop the
    one chunk that actually answers the question (observed failure mode:
    faithfulness/relevancy = 0.0 when compression left too little context).
    If compression leaves fewer than `min_contexts_after_compression` chunks,
    we fall back to the top reranked chunks UNCOMPRESSED rather than risk
    generating from a starved context.
    """
    candidates = hybrid.search(question, top_k=rerank_k, candidate_n=candidate_n)
    reranked = reranker.rerank(question, candidates, top_k=rerank_k)
    compressed = compressor.compress(question, reranked)

    if len(compressed) < min_contexts_after_compression:
        print(
            f"[WARN] Compression left only {len(compressed)} chunk(s) "
            f"(< {min_contexts_after_compression}); falling back to "
            f"uncompressed reranked chunks for this question."
        )
        final_candidates = reranked[:final_k]
    else:
        final_candidates = compressed[:final_k]

    contexts = [r["metadata"].get("text", "") for r in final_candidates if r["metadata"]]
    answer = generate_answer(llm, question, contexts)
    return {"answer": answer, "contexts": contexts}


def run_evaluation(
    judge_llm: ChatGroq,
    testset: List[Dict[str, str]],
    pipeline_fn: Callable[[str], Dict[str, Any]],
    label: str,
) -> pd.DataFrame:
    print(f"\n{'=' * 20} Evaluating: {label} {'=' * 20}")
    rows = []

    for item in testset:
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"[INFO] Running {label} pipeline for: {question}")
        result = pipeline_fn(question)
        answer = result["answer"]
        contexts = result["contexts"] if result["contexts"] else [""]

        print(f"[INFO] Scoring with LLM judge...")
        scores = evaluate_row(judge_llm, question, answer, contexts, ground_truth)

        rows.append({
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            **scores,
        })

    df = pd.DataFrame(rows)
    print(df[["question", "context_precision", "context_recall", "faithfulness", "answer_relevancy"]])

    os.makedirs("eval", exist_ok=True)
    out_path = f"eval/results_{label}.csv"
    df.to_csv(out_path, index=False)
    print(f"[INFO] Saved detailed results to {out_path}")

    print(f"\n--- {label} averages ---")
    print(f"context_precision: {df['context_precision'].mean():.3f}")
    print(f"context_recall:    {df['context_recall'].mean():.3f}")
    print(f"faithfulness:      {df['faithfulness'].mean():.3f}")
    print(f"answer_relevancy:  {df['answer_relevancy'].mean():.3f}")

    return df


if __name__ == "__main__":
    testset = load_testset()

    gen_llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name=LLM_MODEL, temperature=0.0)
    judge_llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name=LLM_MODEL, temperature=0.0)

    store = FaissVectorStore("faiss_store")
    store.load()

    hybrid = HybridRetriever(store)
    hybrid.load_bm25_index()

    reranker = CrossEncoderReranker()
    compressor = ContextualCompressor()

    # --- Baseline: plain FAISS + generate ---
    baseline_df = run_evaluation(
        judge_llm, testset, lambda q: run_baseline_pipeline(store, gen_llm, q), "baseline"
    )

    # --- Advanced: hybrid + rerank + compression + generate ---
    advanced_df = run_evaluation(
        judge_llm,
        testset,
        lambda q: run_advanced_pipeline(hybrid, reranker, compressor, gen_llm, q),
        "advanced",
    )

    # --- Side-by-side comparison ---
    print(f"\n{'=' * 20} Baseline vs Advanced {'=' * 20}")
    for metric in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
        b = baseline_df[metric].mean()
        a = advanced_df[metric].mean()
        delta = a - b
        arrow = "UP" if delta > 0 else ("DOWN" if delta < 0 else "SAME")
        print(f"{metric:20s} baseline={b:.3f}  advanced={a:.3f}  ({arrow} {delta:+.3f})")