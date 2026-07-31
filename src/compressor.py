import os
from typing import List, Dict, Optional, Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

LLM_MODEL = "llama-3.1-8b-instant"

NO_RELEVANT_CONTENT_MARKER = "NO_RELEVANT_CONTENT"


class ContextualCompressor:
    def __init__(self, groq_api_key: Optional[str] = None, model: str = LLM_MODEL):
        self.llm = ChatGroq(
            groq_api_key=groq_api_key or os.getenv("GROQ_API_KEY"),
            model_name=model,
            temperature=0.0,
        )
        self.model = model

    def _compress_single(self, query: str, chunk_text: str) -> Optional[str]:
        system = (
            "You extract the parts of a document chunk that are relevant to "
            "a given query, INCLUDING sentences that provide necessary "
            "supporting/background context for understanding the answer, "
            "even if they don't use the exact same wording as the query. "
            "Copy sentences verbatim - do not paraphrase, summarize, or add "
            "commentary. Be inclusive rather than overly strict: when in "
            "doubt about whether a sentence helps answer the query, keep it. "
            "If NOTHING in the chunk is relevant or helpful for the query, "
            f"respond with exactly: {NO_RELEVANT_CONTENT_MARKER}\n"
            "Otherwise, respond with only the relevant sentence(s), nothing else."
        )
        user = f'Query: "{query}"\n\nChunk:\n"""\n{chunk_text}\n"""'

        response = self.llm.invoke([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
        result = response.content.strip()

        if result == NO_RELEVANT_CONTENT_MARKER or not result:
            return None
        return result

    def compress(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compressed_results = []

        for c in candidates:
            original_text = c["metadata"].get("text", "")
            compressed_text = self._compress_single(query, original_text)

            if compressed_text is None:
                print(f"[INFO] Dropped chunk (no relevant content): {c['metadata'].get('file_name')}")
                continue

            new_c = dict(c)
            new_meta = dict(c["metadata"])
            new_meta["original_text"] = original_text
            new_meta["text"] = compressed_text
            new_c["metadata"] = new_meta
            compressed_results.append(new_c)

        print(f"[INFO] Contextual compression: {len(candidates)} candidates -> {len(compressed_results)} kept")
        return compressed_results


if __name__ == "__main__":
    from src.vectorstore import FaissVectorStore
    from src.hybrid_search import HybridRetriever
    from src.reranker import CrossEncoderReranker

    store = FaissVectorStore("faiss_store")
    store.load()

    hybrid = HybridRetriever(store)
    hybrid.load_bm25_index()

    query = "How does multi-head attention work?"

    candidates = hybrid.search(query, top_k=20, candidate_n=30)
    reranker = CrossEncoderReranker()
    top_results = reranker.rerank(query, candidates, top_k=5)

    print("\n--- Before compression ---")
    for r in top_results:
        print(r["metadata"]["file_name"], "-", r["metadata"]["text"][:150].replace("\n", " "))

    compressor = ContextualCompressor()
    compressed = compressor.compress(query, top_results)

    print("\n--- After compression ---")
    for r in compressed:
        print(r["metadata"]["file_name"], "-", r["metadata"]["text"][:150].replace("\n", " "))
