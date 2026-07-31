import os
import json
import re
from typing import List, Dict, Optional, Any, Tuple

from groq import Groq

from src.vectorstore import FaissVectorStore
from src.hybrid_search import HybridRetriever
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = "llama-3.3-70b-versatile"

# The metadata fields available for self-query filtering. Keep this in sync
# with what data_loader.py actually attaches, or self-query will hallucinate
# fields that don't exist in the index.
KNOWN_METADATA_FIELDS = {
    "file_type": "one of: pdf, txt, csv, xlsx, docx, json",
    "file_name": "the original file name, e.g. 'sample.pdf'",
    "page": "integer page number (PDFs only)",
    "modified_at": "ISO 8601 timestamp of when the source file was last modified",
}


class QueryTransformer:
    def __init__(self, groq_api_key: Optional[str] = None, model: str = LLM_MODEL):
        self.client = Groq(api_key=groq_api_key or os.environ.get("GROQ_API_KEY"))
        self.model = model

    def _chat(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    # ---------- Query Expansion ----------
    def expand_query(self, query: str, n: int = 3) -> List[str]:
        """
        Generate n alternate phrasings of the query to widen recall.
        Returns the original query plus n variants (deduplicated).
        """
        system = (
            "You rewrite search queries to improve retrieval recall. "
            "Given a query, produce alternate phrasings that preserve the "
            "original meaning but use different wording, synonyms, or "
            "specificity levels. Return ONLY a JSON array of strings, "
            "no preamble, no markdown fences."
        )
        user = f'Query: "{query}"\nGenerate {n} alternate phrasings as a JSON array.'

        raw = self._chat(system, user)
        try:
            variants = json.loads(raw)
            if not isinstance(variants, list):
                raise ValueError("Expected a JSON list")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[WARN] Failed to parse query expansion output ({e}); falling back to original query only.")
            variants = []

        all_queries = [query] + [v for v in variants if isinstance(v, str)]
        # de-duplicate while preserving order
        seen = set()
        deduped = []
        for q in all_queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                deduped.append(q)
        return deduped

    # ---------- Multi-Query Retrieval ----------
    def multi_query_retrieve(
        self,
        hybrid_retriever: HybridRetriever,
        query: str,
        top_k: int = 5,
        n_variants: int = 3,
        candidate_n: int = 30,
        metadata_filter: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs retrieval once per query variant, then merges results by summing
        RRF-style contribution across variants (so a chunk retrieved by
        multiple query phrasings ranks higher than one hit by only one).
        """
        queries = self.expand_query(query, n=n_variants)
        print(f"[INFO] Multi-query variants: {queries}")

        combined_scores: Dict[int, float] = {}
        combined_meta: Dict[int, Any] = {}

        for q in queries:
            results = hybrid_retriever.search(
                q, top_k=candidate_n, candidate_n=candidate_n, metadata_filter=metadata_filter
            )
            for r in results:
                idx = r["index"]
                combined_scores[idx] = combined_scores.get(idx, 0.0) + r["rrf_score"]
                combined_meta[idx] = r["metadata"]

        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {"index": idx, "combined_score": score, "metadata": combined_meta[idx]}
            for idx, score in ranked
        ]

    # ---------- Self-Query Retriever ----------
    def parse_self_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Uses the LLM to split a natural-language query into:
          - a semantic_query (what to embed/search for)
          - a metadata_filter dict (structured constraints)

        Example: "find PDF documents about pricing from last month"
          -> semantic_query = "pricing"
          -> metadata_filter = {"file_type": "pdf"}
        """
        fields_desc = "\n".join(f"- {k}: {v}" for k, v in KNOWN_METADATA_FIELDS.items())
        system = (
            "You convert a user's natural language search query into a structured "
            "retrieval request. Extract any explicit metadata constraints the user "
            "mentions, and leave the rest as the semantic search text.\n\n"
            f"Available metadata fields:\n{fields_desc}\n\n"
            "Return ONLY valid JSON in this exact shape, no preamble, no markdown fences:\n"
            '{"semantic_query": "...", "metadata_filter": {}}\n'
            "If no metadata constraints are mentioned, metadata_filter must be an empty object. "
            "Never invent a field that isn't in the list above."
        )
        user = f'Query: "{query}"'

        raw = self._chat(system, user)
        try:
            parsed = json.loads(raw)
            semantic_query = parsed.get("semantic_query", query)
            metadata_filter = parsed.get("metadata_filter", {}) or {}
        except json.JSONDecodeError as e:
            print(f"[WARN] Failed to parse self-query output ({e}); using raw query with no filter.")
            semantic_query, metadata_filter = query, {}

        # Safety net: drop any field the LLM hallucinated that isn't real
        metadata_filter = {k: v for k, v in metadata_filter.items() if k in KNOWN_METADATA_FIELDS}

        print(f"[INFO] Self-query parsed -> semantic_query={semantic_query!r}, metadata_filter={metadata_filter}")
        return semantic_query, metadata_filter

    def self_query_retrieve(
        self,
        hybrid_retriever: HybridRetriever,
        query: str,
        top_k: int = 5,
        candidate_n: int = 30,
    ) -> List[Dict[str, Any]]:
        semantic_query, metadata_filter = self.parse_self_query(query)
        return hybrid_retriever.search(
            semantic_query, top_k=top_k, candidate_n=candidate_n,
            metadata_filter=metadata_filter or None,
        )


# Example usage
if __name__ == "__main__":
    store = FaissVectorStore("faiss_store")
    store.load()

    hybrid = HybridRetriever(store)
    hybrid.load_bm25_index()

    qt = QueryTransformer()

    print("\n=== Query Expansion ===")
    variants = qt.expand_query("attention mechanism")
    print(variants)

    print("\n=== Multi-Query Retrieval ===")
    results = qt.multi_query_retrieve(hybrid, "attention mechanism", top_k=5)
    for r in results:
        print(round(r["combined_score"], 4), r["metadata"]["file_name"], "-", r["metadata"]["text"][:80])

    print("\n=== Self-Query Retriever ===")
    results = qt.self_query_retrieve(hybrid, "find CSV rows about deep learning")
    for r in results:
        print(round(r["rrf_score"], 4), r["metadata"]["file_name"], "-", r["metadata"]["text"][:80])