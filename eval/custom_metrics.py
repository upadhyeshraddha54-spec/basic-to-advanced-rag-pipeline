"""
Custom RAG evaluation metrics, computed via LLM-as-judge instead of the
`ragas` library (which has a broken import on newer langchain-community
versions as of v0.4.3 - see: from langchain_community.chat_models.vertexai
import ChatVertexAI, a module path that was removed upstream).

This implements the same four metrics ragas would give you, with the same
definitions, just scored directly by prompting the LLM to judge each aspect
on a 0.0-1.0 scale. This is a legitimate, widely-used evaluation technique
("LLM-as-judge") and avoids the dependency conflict entirely.
"""
import json
import re
from typing import List, Dict, Any

from langchain_groq import ChatGroq


def _extract_score(raw: str) -> float:
    """Pull a float between 0 and 1 out of the LLM's response, defensively."""
    match = re.search(r"([01](?:\.\d+)?)", raw)
    if match:
        score = float(match.group(1))
        return max(0.0, min(1.0, score))
    print(f"[WARN] Could not parse score from judge output: {raw!r}; defaulting to 0.0")
    return 0.0


def _judge(llm: ChatGroq, system: str, user: str) -> float:
    response = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])
    return _extract_score(response.content.strip())


def context_precision(llm: ChatGroq, question: str, contexts: List[str], ground_truth: str) -> float:
    """
    Of the retrieved contexts, what fraction are actually relevant to
    answering the question? High precision = little irrelevant noise
    retrieved.
    """
    if not contexts or not contexts[0]:
        return 0.0

    system = (
        "You judge retrieval quality for a RAG system. Given a question and a "
        "set of retrieved context chunks, determine what fraction of the "
        "chunks are actually relevant to answering the question. "
        "Respond with ONLY a single number between 0.0 and 1.0, nothing else."
    )
    context_block = "\n---\n".join(contexts)
    user = f"Question: {question}\n\nRetrieved contexts:\n{context_block}\n\nFraction relevant (0.0-1.0):"
    return _judge(llm, system, user)


def context_recall(llm: ChatGroq, question: str, contexts: List[str], ground_truth: str) -> float:
    """
    Does the retrieved context contain everything needed to produce the
    ground truth answer? High recall = nothing important was missed.
    """
    if not contexts or not contexts[0]:
        return 0.0

    system = (
        "You judge retrieval completeness for a RAG system. Given a "
        "ground-truth answer and the context chunks that were retrieved, "
        "determine what fraction of the information in the ground truth "
        "answer is actually supported/present in the retrieved context. "
        "Respond with ONLY a single number between 0.0 and 1.0, nothing else."
    )
    context_block = "\n---\n".join(contexts)
    user = (
        f"Ground truth answer: {ground_truth}\n\n"
        f"Retrieved contexts:\n{context_block}\n\n"
        f"Fraction of ground truth supported by context (0.0-1.0):"
    )
    return _judge(llm, system, user)


def faithfulness(llm: ChatGroq, question: str, answer: str, contexts: List[str]) -> float:
    """
    Is the generated answer actually grounded in the retrieved context, or
    does it contain claims not supported by it (hallucination)? High
    faithfulness = low hallucination.
    """
    context_block = "\n---\n".join(contexts) if contexts and contexts[0] else "(no context provided)"
    system = (
        "You judge factual grounding for a RAG system. Given retrieved "
        "context and a generated answer, determine what fraction of claims "
        "in the answer are actually supported by the context. Claims not "
        "supported by the context count as hallucinations and lower the "
        "score. Respond with ONLY a single number between 0.0 and 1.0, "
        "nothing else."
    )
    user = f"Context:\n{context_block}\n\nGenerated answer: {answer}\n\nFraction of answer supported by context (0.0-1.0):"
    return _judge(llm, system, user)


def answer_relevancy(llm: ChatGroq, question: str, answer: str) -> float:
    """
    Does the generated answer actually address the question that was asked
    (regardless of whether it's factually correct)?
    """
    system = (
        "You judge answer relevancy for a RAG system. Given a question and "
        "a generated answer, determine how directly and completely the "
        "answer addresses the question asked. Respond with ONLY a single "
        "number between 0.0 and 1.0, nothing else."
    )
    user = f"Question: {question}\n\nGenerated answer: {answer}\n\nRelevancy score (0.0-1.0):"
    return _judge(llm, system, user)


def evaluate_row(llm: ChatGroq, question: str, answer: str, contexts: List[str], ground_truth: str) -> Dict[str, float]:
    return {
        "context_precision": context_precision(llm, question, contexts, ground_truth),
        "context_recall": context_recall(llm, question, contexts, ground_truth),
        "faithfulness": faithfulness(llm, question, answer, contexts),
        "answer_relevancy": answer_relevancy(llm, question, answer),
    }