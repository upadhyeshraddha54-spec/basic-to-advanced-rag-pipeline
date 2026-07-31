import os
import faiss
import numpy as np
import pickle
from typing import List, Any, Dict, Optional
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline


class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)

        # Keep the chunk text AND every metadata field from data_loader
        # (file_type, file_name, source_path, modified_at, ingested_at, page, etc.)
        metadatas = []
        for chunk in chunks:
            meta = dict(chunk.metadata)  # copy so we don't mutate the original
            meta["text"] = chunk.page_content
            metadatas.append(meta)

        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

    @staticmethod
    def _matches_filter(meta: Dict, metadata_filter: Dict) -> bool:
        """
        Simple exact-match / membership filter.
        metadata_filter values can be:
          - a scalar -> exact match
          - a list/tuple/set -> value must be one of them
        Example: {"file_type": "pdf"} or {"file_type": ["pdf", "docx"]}
        """
        if not meta:
            return False
        for key, expected in metadata_filter.items():
            actual = meta.get(key)
            if isinstance(expected, (list, tuple, set)):
                if actual not in expected:
                    return False
            else:
                if actual != expected:
                    return False
        return True

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        metadata_filter: Optional[Dict] = None,
        fetch_k: Optional[int] = None,
    ):
        """
        Search the index. If metadata_filter is provided, FAISS's flat index is
        searched with a larger candidate pool (fetch_k), then results are
        filtered down to top_k matches that satisfy the filter.

        FAISS doesn't support native metadata filtering, so this is a
        post-filter approach: over-fetch, filter, truncate.
        """
        total_vectors = self.index.ntotal if self.index is not None else 0
        if total_vectors == 0:
            return []

        if metadata_filter:
            # Over-fetch candidates so we still have enough after filtering.
            # Grow fetch_k progressively if we don't have enough matches yet.
            candidate_k = fetch_k or min(max(top_k * 10, 50), total_vectors)
            candidate_k = min(candidate_k, total_vectors)

            D, I = self.index.search(query_embedding, candidate_k)
            results = []
            for idx, dist in zip(I[0], D[0]):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                meta = self.metadata[idx]
                if self._matches_filter(meta, metadata_filter):
                    results.append({"index": int(idx), "distance": float(dist), "metadata": meta})
                if len(results) >= top_k:
                    break

            if len(results) < top_k and candidate_k < total_vectors:
                print(
                    f"[WARN] Only found {len(results)}/{top_k} results matching filter "
                    f"{metadata_filter} within top {candidate_k} candidates. "
                    f"Consider raising fetch_k."
                )
            return results

        # No filter: original behavior
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if 0 <= idx < len(self.metadata) else None
            results.append({"index": int(idx), "distance": float(dist), "metadata": meta})
        return results

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict] = None,
        fetch_k: Optional[int] = None,
    ):
        """
        query_text: natural language query
        metadata_filter: e.g. {"file_type": "pdf"} or {"file_type": ["pdf", "csv"]}
        fetch_k: how many candidates to pull from FAISS before filtering
                 (only relevant when metadata_filter is set)
        """
        print(f"[INFO] Querying vector store for: '{query_text}' (filter={metadata_filter})")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k, metadata_filter=metadata_filter, fetch_k=fetch_k)


# Example usage
if __name__ == "__main__":
    from src.data_loader import load_all_documents
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()

    # Unfiltered query
    print(store.query("Which technology is used for building intelligent systems?", top_k=3))

    # Filtered query example: only search within PDF-sourced chunks
    print(store.query("Which technology is used for building intelligent systems?", top_k=3, metadata_filter={"file_type": "csv"}))