"""
Async FastAPI gateway around the existing RAG pipelines.

Design notes:
- All the underlying libraries (FAISS, sentence-transformers, cross-encoder,
  ChatGroq's .invoke()) are synchronous/blocking. To keep the event loop
  responsive under concurrent requests, blocking calls are offloaded to a
  thread pool via `run_in_threadpool` rather than awaited directly.
- Models and indexes are loaded ONCE at startup (via the lifespan context
  manager) and reused across requests - reloading a sentence-transformer or
  cross-encoder per-request would be extremely slow.
"""
import os
from contextlib import asynccontextmanager
from typing import Dict, Any, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from langchain_groq import ChatGroq

from src.vectorstore import FaissVectorStore
from src.hybrid_search import HybridRetriever
from src.reranker import CrossEncoderReranker
from src.compressor import ContextualCompressor

from app.schemas import SearchRequest, SearchResponse, RetrievedChunk, HealthResponse

load_dotenv()

GEN_MODEL = "llama-3.3-70b-versatile"

# Populated at startup, reused across requests
state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] Loading FAISS index, BM25 index, reranker, and LLM...")

    store = FaissVectorStore("faiss_store")
    store.load()

    hybrid = HybridRetriever(store)
    hybrid.load_bm25_index()

    reranker = CrossEncoderReranker()
    compressor = ContextualCompressor()
    llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name=GEN_MODEL, temperature=0.0)

    state["store"] = store
    state["hybrid"] = hybrid
    state["reranker"] = reranker
    state["compressor"] = compressor
    state["llm"] = llm

    print("[STARTUP] Ready.")
    yield
    state.clear()


app = FastAPI(
    title="Advanced RAG API",
    description="Baseline vs. advanced (hybrid search + rerank + compression) RAG endpoints",
    version="1.0.0",
    lifespan=lifespan,
)


def _generate_answer(llm: ChatGroq, question: str, contexts: List[str]) -> str:
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


def _run_baseline(query: str, top_k: int, metadata_filter) -> Dict[str, Any]:
    store: FaissVectorStore = state["store"]
    llm: ChatGroq = state["llm"]

    results = store.query(query, top_k=top_k, metadata_filter=metadata_filter)
    contexts = [r["metadata"] for r in results if r["metadata"]]
    texts = [c.get("text", "") for c in contexts]

    answer = _generate_answer(llm, query, texts)
    return {"answer": answer, "contexts": contexts}


def _run_advanced(
    query: str,
    top_k: int,
    metadata_filter,
    candidate_n: int = 40,
    rerank_k: int = 15,
    min_contexts_after_compression: int = 3,
) -> Dict[str, Any]:
    hybrid: HybridRetriever = state["hybrid"]
    reranker: CrossEncoderReranker = state["reranker"]
    compressor: ContextualCompressor = state["compressor"]
    llm: ChatGroq = state["llm"]

    candidates = hybrid.search(query, top_k=rerank_k, candidate_n=candidate_n, metadata_filter=metadata_filter)
    reranked = reranker.rerank(query, candidates, top_k=rerank_k)
    compressed = compressor.compress(query, reranked)

    if len(compressed) < min_contexts_after_compression:
        final_candidates = reranked[:top_k]
    else:
        final_candidates = compressed[:top_k]

    contexts = [r["metadata"] for r in final_candidates if r["metadata"]]
    texts = [c.get("text", "") for c in contexts]

    answer = _generate_answer(llm, query, texts)
    return {"answer": answer, "contexts": contexts}


def _to_response(query: str, pipeline: str, result: Dict[str, Any]) -> SearchResponse:
    chunks = [
        RetrievedChunk(
            file_name=c.get("file_name"),
            file_type=c.get("file_type"),
            text=c.get("text", ""),
        )
        for c in result["contexts"]
    ]
    return SearchResponse(
        query=query,
        answer=result["answer"],
        pipeline=pipeline,
        contexts=chunks,
        num_contexts_used=len(chunks),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if state else "starting",
        baseline_ready="store" in state,
        advanced_ready="hybrid" in state and "reranker" in state and "compressor" in state,
    )


@app.post("/v1/search/baseline", response_model=SearchResponse)
async def search_baseline(req: SearchRequest):
    """Plain FAISS similarity search -> generate. No hybrid, rerank, or compression."""
    if "store" not in state:
        raise HTTPException(status_code=503, detail="Index not loaded yet")

    result = await run_in_threadpool(
        _run_baseline, req.query, req.top_k, req.metadata_filter
    )
    return _to_response(req.query, "baseline", result)


@app.post("/v1/search/advanced", response_model=SearchResponse)
async def search_advanced(req: SearchRequest):
    """Query Transformation-ready slot -> RRF Hybrid Search -> Cross-Encoder Reranking -> Contextual Compression -> generate."""
    if "hybrid" not in state:
        raise HTTPException(status_code=503, detail="Index not loaded yet")

    result = await run_in_threadpool(
        _run_advanced, req.query, req.top_k, req.metadata_filter
    )
    return _to_response(req.query, "advanced", result)