from typing import List, Dict, Optional, Any

from sentence_transformers import CrossEncoder

from src.vectorstore import FaissVectorStore
from src.hybrid_search import HybridRetriever


class CrossEncoderReranker:
    """
    Re-ranks a candidate pool of chunks using a cross-encoder.

    Why this matters: FAISS/BM25/RRF all score query and document
    INDEPENDENTLY (bi-encoder style) - the query embedding and chunk embedding
    are computed separately, then compared. A cross-encoder instead feeds the
    (query, chunk) pair TOGETHER into one model, so it can directly reason
    about how well they match. This is much more accurate but far more
    expensive, which is why it's only run on a small candidate pool
    (e.g. top 20-30 from hybrid search) rather than the whole corpus.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)
        print(f"[INFO] Loaded cross-encoder re-ranker: {model_name}")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        candidates: list of dicts with at least {"metadata": {"text": ...}, ...}
                    (the output shape of HybridRetriever.search / FaissVectorStore.search)
        Returns the same dicts, re-sorted by cross-encoder relevance score,
        truncated to top_k, with a "rerank_score" field added.
        """
        if not candidates:
            return []

        pairs = [(query, c["metadata"]["text"]) for c in candidates]
        scores = self.model.predict(pairs)  # higher = more relevant

        for c, score in zip(candidates, scores):
            c["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        return reranked[:top_k]


# Example usage
if __name__ == "__main__":
    from src.data_loader import load_all_documents

    store = FaissVectorStore("faiss_store")
    store.load()  # assumes hybrid_search.py has already been run and index exists

    hybrid = HybridRetriever(store)
    hybrid.load_bm25_index()

    query = "attention mechanism"

    # Get a wider candidate pool from hybrid search before reranking
    candidates = hybrid.search(query, top_k=20, candidate_n=30)
    print(f"\n--- Hybrid search candidates (top 20, pre-rerank) ---")
    for c in candidates[:5]:
        print(round(c["rrf_score"], 4), c["metadata"]["file_name"], "-", c["metadata"]["text"][:80])

    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(query, candidates, top_k=5)

    print(f"\n--- After cross-encoder re-ranking (top 5) ---")
    for r in reranked:
        print(round(r["rerank_score"], 4), r["metadata"]["file_name"], "-", r["metadata"]["text"][:80])