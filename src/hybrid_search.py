import os
import pickle
import re
from typing import List, Dict, Optional, Any

from rank_bm25 import BM25Okapi

from src.vectorstore import FaissVectorStore


def _tokenize(text: str) -> List[str]:
    """Simple, dependency-free tokenizer: lowercase + split on non-alphanumeric."""
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    """
    Combines BM25 (keyword/lexical search) with FAISS (dense vector search)
    using Reciprocal Rank Fusion (RRF) to merge the two ranked lists.

    Why RRF instead of averaging raw scores: BM25 scores and FAISS L2 distances
    live on completely different scales, so combining them directly is meaningless.
    RRF only cares about each result's *rank* in each list, which makes it a
    scale-free way to merge them.
    """

    def __init__(self, vectorstore: FaissVectorStore, persist_dir: str = "faiss_store", rrf_k: int = 60):
        self.vectorstore = vectorstore
        self.persist_dir = persist_dir
        self.rrf_k = rrf_k  # standard RRF constant; higher = flatter weighting of rank position
        self.bm25 = None
        self.tokenized_corpus = None

    def build_bm25_index(self):
        """Build a BM25 index over the same chunks stored in the FAISS metadata."""
        if not self.vectorstore.metadata:
            raise ValueError("Vectorstore metadata is empty. Build or load the FAISS index first.")

        texts = [meta.get("text", "") for meta in self.vectorstore.metadata]
        self.tokenized_corpus = [_tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"[INFO] Built BM25 index over {len(texts)} chunks.")
        self.save_bm25_index()

    def save_bm25_index(self):
        path = os.path.join(self.persist_dir, "bm25.pkl")
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "tokenized_corpus": self.tokenized_corpus}, f)
        print(f"[INFO] Saved BM25 index to {path}")

    def load_bm25_index(self):
        path = os.path.join(self.persist_dir, "bm25.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.tokenized_corpus = data["tokenized_corpus"]
        print(f"[INFO] Loaded BM25 index from {path}")

    def _bm25_ranked_indices(self, query: str, top_n: int) -> List[int]:
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[:top_n]

    def _faiss_ranked_indices(self, query: str, top_n: int, metadata_filter: Optional[Dict] = None) -> List[int]:
        results = self.vectorstore.query(query, top_k=top_n, metadata_filter=metadata_filter)
        return [r["index"] for r in results]

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_n: int = 30,
        metadata_filter: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top_k results by fusing BM25 and FAISS rankings.

        candidate_n: how many results to pull from EACH method before fusing.
                     Higher = more thorough fusion but slower.
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not built/loaded. Call build_bm25_index() or load_bm25_index() first.")

        bm25_ranked = self._bm25_ranked_indices(query, candidate_n)
        faiss_ranked = self._faiss_ranked_indices(query, candidate_n, metadata_filter=metadata_filter)

        # Reciprocal Rank Fusion: score = sum over each list of 1 / (rrf_k + rank)
        rrf_scores: Dict[int, float] = {}

        for rank, idx in enumerate(bm25_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        for rank, idx in enumerate(faiss_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        # If a metadata_filter was applied, FAISS results already respect it.
        # BM25 has no notion of metadata, so re-apply the filter to BM25-only hits.
        if metadata_filter:
            filtered_scores = {}
            for idx, score in rrf_scores.items():
                if idx >= len(self.vectorstore.metadata):
                    continue
                meta = self.vectorstore.metadata[idx]
                if FaissVectorStore._matches_filter(meta, metadata_filter):
                    filtered_scores[idx] = score
            rrf_scores = filtered_scores

        ranked_final = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, score in ranked_final:
            meta = self.vectorstore.metadata[idx] if idx < len(self.vectorstore.metadata) else None
            results.append({"index": idx, "rrf_score": score, "metadata": meta})
        return results


# Example usage
if __name__ == "__main__":
    from src.data_loader import load_all_documents

    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()

    hybrid = HybridRetriever(store)
    hybrid.build_bm25_index()

    print("\n--- Hybrid search (no filter) ---")
    for r in hybrid.search("attention mechanism", top_k=3):
        print(r["rrf_score"], r["metadata"]["file_name"], "-", r["metadata"]["text"][:80])

    print("\n--- Hybrid search (filtered to CSV) ---")
    for r in hybrid.search("attention mechanism", top_k=3, metadata_filter={"file_type": "csv"}):
        print(r["rrf_score"], r["metadata"]["file_name"], "-", r["metadata"]["text"][:80])