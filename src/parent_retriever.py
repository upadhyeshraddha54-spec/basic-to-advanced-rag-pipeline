import os
import pickle
import uuid
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ParentDocumentRetriever:
    """
    Two-level chunking retriever:

      - PARENT chunks: larger pieces of text (e.g. 2000 chars) that preserve
        full context/nuance.
      - CHILD chunks: smaller pieces (e.g. 400 chars) cut from each parent,
        used ONLY for embedding/search precision.

    Why: small chunks embed more precisely (a 400-char chunk about
    "multi-head attention" has a sharper, less diluted embedding than a
    2000-char chunk covering five subtopics), which improves search accuracy.
    But small chunks alone often lack enough surrounding context for the LLM
    to generate a good answer. So we search on children, then hand back their
    parent for generation - precision at search time, context at generation
    time.
    """

    def __init__(
        self,
        persist_dir: str = "faiss_store_parent",
        embedding_model: str = "all-MiniLM-L6-v2",
        parent_chunk_size: int = 2000,
        parent_chunk_overlap: int = 200,
        child_chunk_size: int = 400,
        child_chunk_overlap: int = 50,
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.model = SentenceTransformer(embedding_model)
        print(f"[INFO] Loaded embedding model: {embedding_model}")

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size, chunk_overlap=parent_chunk_overlap
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap
        )

        self.index = None
        self.child_metadata: List[Dict[str, Any]] = []   # one entry per FAISS vector
        self.parent_store: Dict[str, Dict[str, Any]] = {}  # parent_id -> {"text": ..., "metadata": ...}

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building parent-document index from {len(documents)} raw documents...")

        all_child_texts = []
        all_child_meta = []

        for doc in documents:
            parent_chunks = self.parent_splitter.split_text(doc.page_content)

            for parent_text in parent_chunks:
                parent_id = str(uuid.uuid4())
                self.parent_store[parent_id] = {
                    "text": parent_text,
                    "metadata": dict(doc.metadata),
                }

                child_texts = self.child_splitter.split_text(parent_text)
                for child_text in child_texts:
                    all_child_texts.append(child_text)
                    child_meta = dict(doc.metadata)
                    child_meta["parent_id"] = parent_id
                    child_meta["text"] = child_text
                    all_child_meta.append(child_meta)

        print(f"[INFO] Created {len(self.parent_store)} parent chunks -> {len(all_child_texts)} child chunks")

        print(f"[INFO] Embedding {len(all_child_texts)} child chunks...")
        embeddings = self.model.encode(all_child_texts, show_progress_bar=True).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.child_metadata = all_child_meta

        self.save()
        print(f"[INFO] Parent-document index built and saved to {self.persist_dir}")

    def save(self):
        faiss.write_index(self.index, os.path.join(self.persist_dir, "faiss.index"))
        with open(os.path.join(self.persist_dir, "child_metadata.pkl"), "wb") as f:
            pickle.dump(self.child_metadata, f)
        with open(os.path.join(self.persist_dir, "parent_store.pkl"), "wb") as f:
            pickle.dump(self.parent_store, f)

    def load(self):
        self.index = faiss.read_index(os.path.join(self.persist_dir, "faiss.index"))
        with open(os.path.join(self.persist_dir, "child_metadata.pkl"), "rb") as f:
            self.child_metadata = pickle.load(f)
        with open(os.path.join(self.persist_dir, "parent_store.pkl"), "rb") as f:
            self.parent_store = pickle.load(f)
        print(f"[INFO] Loaded parent-document index from {self.persist_dir} "
              f"({len(self.parent_store)} parents, {len(self.child_metadata)} children)")

    def query(self, query_text: str, top_k: int = 5, child_fetch_k: int = 15) -> List[Dict[str, Any]]:
        """
        Searches on CHILD chunks (precise), then returns their unique PARENT
        chunks (context-rich), deduplicated and ordered by best child match.
        """
        query_emb = self.model.encode([query_text]).astype("float32")
        D, I = self.index.search(query_emb, min(child_fetch_k, self.index.ntotal))

        seen_parents = set()
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx < 0 or idx >= len(self.child_metadata):
                continue
            child_meta = self.child_metadata[idx]
            parent_id = child_meta["parent_id"]

            if parent_id in seen_parents:
                continue  # already returned this parent from a better-ranked child
            seen_parents.add(parent_id)

            parent = self.parent_store[parent_id]
            results.append({
                "parent_id": parent_id,
                "distance": float(dist),
                "matched_child_text": child_meta["text"],
                "parent_text": parent["text"],
                "metadata": parent["metadata"],
            })

            if len(results) >= top_k:
                break

        return results


# Example usage
if __name__ == "__main__":
    from src.data_loader import load_all_documents

    docs = load_all_documents("data")
    retriever = ParentDocumentRetriever()
    retriever.build_from_documents(docs)
    retriever.load()

    query = "How does multi-head attention work?"
    results = retriever.query(query, top_k=3)

    print(f"\n--- Parent Document Retriever results for: '{query}' ---")
    for r in results:
        print(f"\n[matched child]: {r['matched_child_text'][:100]}")
        print(f"[returned parent, {len(r['parent_text'])} chars]: {r['parent_text'][:300]}...")