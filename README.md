# Basic to Advanced RAG Pipeline

## Overview

This project shows how a basic Retrieval-Augmented Generation (RAG) system can be improved using modern retrieval techniques.

It starts with a simple FAISS-based RAG pipeline and adds techniques one at a time: Metadata Filtering, Hybrid Search, Cross-Encoder Re-ranking, Query Transformation, Contextual Compression, and Parent Document Retrieval. The baseline and advanced pipelines are then evaluated side by side using RAGAS-style metrics.

The goal is to understand how each technique actually affects retrieval and answer quality, not just to add features.

---

## Features

- Multi-format document ingestion (PDF, TXT, CSV, DOCX, XLSX, JSON)
- FAISS vector database
- Metadata filtering
- Hybrid search (FAISS + BM25)
- Cross-encoder re-ranking
- Query expansion
- Multi-query retrieval
- Self-query retriever
- Contextual compression
- Parent document retrieval
- Groq Llama-based answer generation
- Custom RAG evaluation (LLM-as-judge)

---

## Baseline vs. Advanced

| Capability | Baseline | Advanced |
|---|:---:|:---:|
| Vector search | ✅ | ✅ |
| Metadata filtering | ❌ | ✅ |
| Hybrid search (BM25 + FAISS) | ❌ | ✅ |
| Cross-encoder re-ranking | ❌ | ✅ |
| Query expansion / multi-query | ❌ | ✅ |
| Self-query retriever | ❌ | ✅ |
| Contextual compression | ❌ | ✅ |
| Parent document retrieval | ❌ | ✅ |
| Formal evaluation | ❌ | ✅ |

---

## Tech Stack

| Technology | Purpose |
|---|---|
| LangChain | Document loading and retriever interfaces |
| FAISS | Vector similarity search |
| sentence-transformers (`all-MiniLM-L6-v2`) | Embeddings |
| rank-bm25 | Keyword search |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | Re-ranking |
| Groq (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`) | Generation |
| pandas | Evaluation results |

---

## Architecture

```
User Query
      │
      ▼
Query Transformation
      │
      ▼
Hybrid Retrieval (FAISS + BM25)
      │
      ▼
Cross-Encoder Re-ranking
      │
      ▼
Contextual Compression
      │
      ▼
Groq Llama Generation
      │
      ▼
Final Answer
```

Parent Document Retrieval is a separate retrieval strategy with its own FAISS index (`faiss_store_parent/`), rather than a stage in the main pipeline above — it's used and evaluated independently.

---

## Baseline Pipeline

1. Load documents (PDF, TXT, CSV, DOCX, XLSX, JSON)
2. Chunk text (`chunk_size=1000`, `chunk_overlap=200`)
3. Embed chunks with `all-MiniLM-L6-v2`
4. Store vectors in FAISS
5. Retrieve top-k similar chunks
6. Generate an answer with Groq's `llama-3.3-70b-versatile`

Works, but relies only on vector similarity — no filtering, no reranking, no query understanding.

---

## RAG Enhancements

### Metadata Filtering

Added metadata — file name, file type, source path, size, and timestamps — to every document chunk during loading (`src/data_loader.py`). Since FAISS doesn't support metadata filtering natively, a larger candidate set is retrieved first and then filtered down to the requested `top_k`.

This enables queries such as:
- Search only PDFs
- Search only CSV files

### Hybrid Search (BM25 + FAISS)

Combines semantic search (FAISS) with keyword search (BM25) over the same chunk text (`src/hybrid_search.py`). Results are merged using Reciprocal Rank Fusion (RRF), which is scale-free — important because BM25 scores and FAISS distances aren't directly comparable. For "attention mechanism," this surfaced a relevant CSV row that pure vector search alone ranked lower.

### Cross-Encoder Re-ranking

Hybrid search scores the query and each chunk independently, which misses nuance. The top ~20–30 hybrid candidates get re-scored by a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) that looks at the query and chunk together. In testing, this correctly demoted chunks that matched on keywords but turned out to be generic filler text rather than real explanations.

### Query Expansion

The LLM generates alternate phrasings of a query (e.g. "attention mechanism" → "focus mechanism," "concentration technique"), and each variant is searched separately (`src/query_transform.py`). This widens recall for queries phrased differently than the corpus's own wording.

### Multi-Query Retrieval

Results from every phrasing produced by Query Expansion are merged by summing RRF-style scores across variants, so a chunk that multiple phrasings agree on ranks higher than one only a single phrasing matched.

### Self-Query Retriever

Lets a question like *"find CSV rows about deep learning"* get parsed automatically into a semantic search string (`"deep learning"`) plus a metadata filter (`{"file_type": "csv"}`), instead of requiring the filter to be specified manually. A safety net drops any field the LLM hallucinates that isn't a real metadata field. Test queries parsed correctly, and every result genuinely came from the requested file type.

### Contextual Compression

Trims each retrieved chunk down to the sentences relevant to the query, dropping the whole chunk if nothing in it is relevant (`src/compressor.py`). An early version of this was too strict — it dropped a chunk that explained self-attention because it didn't literally contain the phrase "regular attention," and the model answered "I don't know" instead of correctly. The fix was to loosen the compression prompt to keep supporting context, and add a fallback to the uncompressed reranked chunks if compression leaves too few.

### Parent Document Retriever

Small chunks embed precisely but lack context; large chunks give context but dilute the embedding. This splits documents into large parent chunks (~2000 chars) and smaller child chunks (~400 chars) (`src/parent_retriever.py`) — only children are embedded and searched, and when one matches, its parent is returned. One test surfaced a real limitation: the best-matching child chunk was a heading sitting right at the boundary between two parents, so the wrong adjacent parent section came back — a known tradeoff of character-count-based chunking versus splitting along document structure.

---

## Evaluation

The baseline and advanced pipelines were evaluated on a 7-question hand-crafted test set (`eval/qa_testset.json`) using a custom LLM-as-judge implementation of four RAGAS-style metrics — Context Precision, Context Recall, Faithfulness, and Answer Relevancy. A custom evaluator (`eval/custom_metrics.py`) was built due to a compatibility issue with the current `ragas` version.

**Final results:**

| Metric | Baseline | Advanced | Change |
|---|---|---|---|
| Context Precision | 0.574 | **0.679** | **+0.105 (meaningfully better)** |
| Context Recall | 0.810 | 0.737 | −0.073 (expected precision/recall tradeoff) |
| Faithfulness | 0.896 | 0.894 | ≈ unchanged |
| Answer Relevancy | 0.936 | 0.921 | ≈ unchanged |

**How to read this honestly:** the advanced pipeline delivers meaningfully cleaner retrieval
(higher precision) without sacrificing faithfulness or answer relevancy (both differences are
within noise). The recall dip is a real, expected tradeoff — aggressive compression and reranking
narrow the context down, which occasionally trims something that would have added completeness,
even though it doesn't hurt the final answer's grounding. This is a far more credible result than
a flat "everything got better" — it shows the specific tradeoff advanced retrieval techniques
actually make.

**A rate-limit lesson worth mentioning too:** the evaluation makes many LLM calls (generation +
4 judge calls × 7 questions × 2 pipelines, plus one compression call per candidate chunk). This
exhausted Groq's free-tier daily token quota for `llama-3.3-70b-versatile` mid-run more than once.
The fix was to move compression, generation, and judging over to the smaller, separately-quota'd
`llama-3.1-8b-instant` model for evaluation runs — a practical lesson in managing LLM API costs
during development.

---

## Project Structure

```
rag/
├── src/
│   ├── data_loader.py        # Loads PDF/TXT/CSV/DOCX/XLSX/JSON, tags metadata
│   ├── vectorstore.py        # FAISS store: build, save, load, query, metadata filtering
│   ├── hybrid_search.py      # BM25 + FAISS fusion via Reciprocal Rank Fusion
│   ├── reranker.py           # Cross-encoder re-ranking
│   ├── query_transform.py    # Query expansion, multi-query, self-query retriever
│   ├── compressor.py         # Contextual compression
│   ├── parent_retriever.py   # Parent/child two-level chunking retriever
│   ├── embedding.py          # Chunking + embedding pipeline
│   └── search.py             # Original baseline RAG search + Groq summarization
├── eval/
│   ├── qa_testset.json       # 7 hand-crafted Q&A pairs with ground truth
│   ├── custom_metrics.py     # LLM-as-judge implementation of the 4 RAGAS-style metrics
│   ├── run_ragas.py          # Runs baseline vs. advanced pipeline, evaluates both
│   ├── results_baseline.csv  # Per-question scores, baseline pipeline
│   └── results_advanced.csv  # Per-question scores, advanced pipeline
├── data/                     # Source documents
│   ├── pdf/
│   ├── text_files/
│   └── csv/
├── faiss_store/              # Main FAISS index + metadata + BM25 index
├── faiss_store_parent/       # Parent-document retriever's separate index
├── requirements.txt
└── .env                      # GROQ_API_KEY (not committed)
```

---

## Setup

```bash
git clone https://github.com/upadhyeshraddha54-spec/rag.git
cd rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install rank-bm25 langchain-text-splitters pandas
```

Create a `.env` file in the root:
```
GROQ_API_KEY=your_groq_api_key_here
```

**Important:** every module lives inside `src/` or `eval/` and uses package-relative imports. Always
run modules from the repo root using the `-m` flag, never by path directly:

```bash
python -m src.data_loader
python -m src.vectorstore
python -m src.hybrid_search
python -m src.reranker
python -m src.query_transform
python -m src.compressor
python -m src.parent_retriever
python -m eval.run_ragas
```

---

## Key Learnings

- **"Advanced" doesn't automatically mean "better."** Every technique we added had to be tuned —
  contextual compression in particular could actively hurt results if configured too aggressively.
- **Metadata quality matters as much as the retrieval algorithm.** Self-query retrieval is only as
  good as the metadata fields it can filter on.
- **Test data quality matters.** Early evaluation runs on a query ("attention mechanism") the
  corpus had no real content for produced misleading results across every technique — a data
  problem, not a pipeline problem. Adding one real, accurate reference document fixed this.
- **Test *question* wording matters too.** A ground-truth question using terminology the corpus
  never uses ("regular attention") caused a false catastrophic failure that looked like a pipeline
  bug but was actually a corpus/question mismatch combined with an overly strict generation prompt.
- **Dependency issues are part of real ML engineering.** The `ragas` library's broken import wasn't
  a dead end — building a custom LLM-as-judge evaluator instead was a legitimate (and arguably
  better) solution.
- **Watch your API token budget during evaluation.** Comparing two pipelines across many metrics
  and questions adds up fast; a smaller model for high-volume calls (compression, judging) avoids
  hitting rate limits mid-experiment.

---

## Pushing to GitHub

If this is the first time pushing (repo not yet initialized):

```bash
git init
git add .
git commit -m "Advanced RAG: hybrid search, reranking, query transformation, compression, parent retrieval, evaluation"
git branch -M main
git remote add origin https://github.com/upadhyeshraddha54-spec/rag.git
git push -u origin main
```

If the repo already exists and is already connected to GitHub (as this one is), just commit and push
the new work:

```bash
git add .
git commit -m "Add hybrid search, reranking, query transformation, compression, parent retriever, and evaluation"
git push
```

**Before committing, make sure these are excluded** (add to `.gitignore` if not already there):
```
.env
.venv/
__pycache__/
faiss_store/
faiss_store_parent/
*.pkl
```
The FAISS indexes and pickled metadata/BM25 files can be large and are regenerable from `data/` —
they don't need to live in version control. Your `.env` should never be committed since it contains
your API key.

After pushing, double check on GitHub that:
- `README.md` renders correctly
- `.env` is NOT visible in the repository (if it is, rotate your Groq API key immediately and add
  `.env` to `.gitignore` retroactively)
- The `eval/results_baseline.csv` and `eval/results_advanced.csv` files are present, since they're
  good evidence of real evaluation work
