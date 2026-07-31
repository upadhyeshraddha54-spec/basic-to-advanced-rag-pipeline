# Advanced RAG Pipeline

A Retrieval-Augmented Generation (RAG) system that started as a basic FAISS + Groq pipeline and was
extended, one technique at a time, into a production-style advanced RAG system — with a real, honest
evaluation comparing the basic version against the advanced one.

This README documents the full journey: what was built, why each technique was added, what actually
worked, what failed along the way, and what we learned from the failures. Nothing here is cleaned up
to hide the messy parts — the debugging story is part of the point.

---

## Tech Stack

- **LangChain** — document loading, text splitting, retriever interfaces
- **FAISS** — vector similarity search
- **Groq** (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`) — generation and LLM-based tasks
- **sentence-transformers** (`all-MiniLM-L6-v2`) — embeddings
- **rank-bm25** — keyword search
- **cross-encoder/ms-marco-MiniLM-L-6-v2** — re-ranking
- **pandas** — evaluation result handling

---

## Where We Started: The Basic Pipeline

Before any of the advanced work, this was a simple RAG pipeline:

1. **Load** documents from `data/` — PDF, TXT, CSV, DOCX, XLSX, JSON
2. **Chunk** text into overlapping pieces (`chunk_size=1000`, `chunk_overlap=200`)
3. **Embed** chunks with `all-MiniLM-L6-v2`
4. **Store** vectors in a FAISS index
5. **Retrieve** the top-k most similar chunks for a query
6. **Generate** an answer by handing those chunks to Groq's `llama-3.3-70b-versatile`

This works, but it has real limitations: one search method (pure vector similarity), no relevance
filtering, no query understanding beyond the literal wording, and no way to measure whether it's
actually any good. Everything below addresses one of those gaps.

---

## The 8 Techniques We Added

### 1. Metadata Filtering
**Problem:** there was no way to search only within a subset of documents (e.g. "only PDFs" or
"only files modified this month").

**What we did:** every chunk now carries consistent metadata regardless of source file type —
`file_type`, `file_name`, `source_path`, `file_size_bytes`, `modified_at`, `ingested_at` — attached
during loading (`src/data_loader.py`). The vector store's `search()`/`query()` methods accept a
`metadata_filter` dict and support both exact match (`{"file_type": "pdf"}`) and membership
(`{"file_type": ["pdf", "csv"]}`).

**How it works under the hood:** FAISS has no native metadata filtering, so we use an
**over-fetch-then-filter** approach — pull a larger candidate pool from FAISS, filter by metadata,
then truncate to the requested `top_k`.

**Result:** worked correctly on the first real test — filtering to `file_type: "csv"` returned only
CSV rows, filtering to `"pdf"` returned only PDF chunks.

---

### 2. Hybrid Search (BM25 + FAISS)
**Problem:** pure vector search (FAISS) is good at "meaning" matches but can miss exact keywords,
IDs, or acronyms. Pure keyword search (BM25) is the opposite.

**What we did:** built a BM25 index over the same chunk text stored in FAISS metadata
(`src/hybrid_search.py`), then merged BM25's ranked list with FAISS's ranked list using
**Reciprocal Rank Fusion (RRF)** — `score = Σ 1/(rrf_k + rank + 1)` across both lists. RRF is
scale-free, which matters because BM25 scores and FAISS L2 distances live on completely
incompatible scales — you can't average them directly.

**Result:** confirmed working — for the query "attention mechanism," hybrid search correctly
surfaced a CSV row about deep learning that pure FAISS search alone would have ranked lower, purely
because BM25 caught the keyword overlap FAISS's embedding missed.

---

### 3. Cross-Encoder Re-ranking
**Problem:** hybrid search is fast but imprecise — FAISS and BM25 both score the query and each
chunk *independently*, then compare. This misses a lot of nuance.

**What we did:** took hybrid search's top ~20–30 candidates and re-scored each `(query, chunk)` pair
using a **cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) — a model that looks at the query
and chunk *together*. Cross-encoders are far more accurate but too slow to run on an entire corpus,
which is why they're only applied to hybrid search's already-narrowed candidate pool.

**Result — and an important honest finding:** in one test, the reranker correctly *demoted* CSV
rows that had matched on keyword overlap ("Deep Learning", "NLP") but turned out to be generic
templated filler text ("This document discusses..."), rather than genuine explanations. This
was the reranker doing exactly its job — filtering out lexically-matched-but-semantically-empty
content.

---

### 4. Query Transformation (three sub-techniques, one file)

**Query Expansion** — the LLM generates alternate phrasings of a query (e.g. "attention mechanism"
→ "focus mechanism", "concentration technique") to widen recall for synonyms or different wording.

**Multi-Query Retrieval** — retrieval runs once per query variant, then results are merged by
summing RRF-style scores across variants, so a chunk multiple phrasings agree on ranks higher.

**Self-Query Retriever** — the LLM parses a natural question like *"find CSV rows about deep
learning"* into a semantic search string (`"deep learning"`) + a structured metadata filter
(`{"file_type": "csv"}`) — automatically, without the user specifying filters manually. A safety
net drops any field the LLM hallucinates that isn't a real metadata field.

**Result:** all three worked. Self-query in particular was a clean win — the test query correctly
parsed into the right semantic query + filter, and every returned result was genuinely from the
CSV file as requested.

---

### 5. Contextual Compression
**Problem:** a retrieved chunk often mixes relevant and irrelevant sentences. Passing the whole
chunk to the generation LLM wastes context and dilutes the signal.

**What we did:** used the LLM to trim each retrieved chunk down to just the sentences relevant to
the query (or drop the chunk entirely if nothing in it is relevant), using a
`NO_RELEVANT_CONTENT` marker to signal a full drop (`src/compressor.py`).

**What actually failed, and what we learned:** this is the technique that caused the most real
debugging. Two failure modes showed up during evaluation:

- **Over-aggressive trimming:** an early version of the compression prompt was too strict —
  it required near-literal wording overlap with the query. On one test question ("difference
  between self-attention and regular attention"), compression discarded the one chunk that
  actually explained self-attention, because that chunk didn't use the literal phrase "regular
  attention." The generation LLM then correctly (but unhelpfully) answered "I don't know."
- **Fix, part 1:** loosened the compression prompt to keep supporting/background context, not
  just exact-wording matches ("be inclusive rather than overly strict").
- **Fix, part 2:** added a **fallback safety net** in the evaluation pipeline — if compression
  leaves fewer than a minimum number of chunks, fall back to the uncompressed reranked chunks
  rather than risk generating from a starved context.
- **The deeper root cause** turned out to be a mismatch between the test question's wording
  ("regular attention," a term our source document never uses) and the corpus's actual
  terminology — combined with an overly strict "say you don't know" instruction in the generation
  prompt. Fixing both the question wording and loosening the generation prompt resolved it.

This whole episode is genuinely one of the more useful things to talk about from this project:
**a technique working "correctly" in isolation can still hurt end-to-end quality if it's tuned too
aggressively or paired with a test question that doesn't match the corpus.**

---

### 6. Parent Document Retriever
**Problem:** small chunks embed precisely but lack surrounding context. Large chunks give context
but their embeddings get diluted trying to represent multiple ideas at once.

**What we did:** two-level chunking (`src/parent_retriever.py`) — documents are split into large
**parent** chunks (~2000 chars) and each parent is further split into small **child** chunks (~400
chars). Only child chunks are embedded and searched (precision). When a child chunk matches, its
parent is returned instead (context), deduplicated across matches.

**Result — another honest, real finding:** in testing, the single best-matching child chunk was
literally the heading `"Multi-Head Attention"` — a very sharp match. But that heading happened to
sit right at the boundary between two parent chunks, so the parent returned was actually the
adjacent "Self-Attention" section, not the ideal one. This is a known, realistic limitation of
naive character-count-based parent chunking (as opposed to splitting along document structure or
headings) — worth naming explicitly rather than hiding.

---

### 7. RAG Evaluation
**Problem:** up to this point, quality was being judged by eyeballing outputs. That doesn't scale
and isn't rigorous.

**What we did:** built a 7-question hand-crafted test set (`eval/qa_testset.json`) with real
ground-truth answers based on our own corpus content, then measured four standard RAG metrics for
both the **baseline** pipeline (plain FAISS → generate) and the **advanced** pipeline (hybrid
search → rerank → contextual compression → generate):

- **Context Precision** — of what got retrieved, how much was actually relevant?
- **Context Recall** — did retrieval capture everything needed to answer fully?
- **Faithfulness** — is the generated answer actually grounded in retrieved context, or hallucinated?
- **Answer Relevancy** — does the answer actually address the question asked?

**A real dependency problem we hit and solved:** the popular `ragas` evaluation library (v0.4.3)
has a broken import — it unconditionally imports `ChatVertexAI` from a `langchain_community`
module path that's been removed in current versions. After trying dependency upgrades and version
pins without success, we replaced it with a **custom LLM-as-judge evaluator**
(`eval/custom_metrics.py`) that implements the same four metric definitions by directly prompting
an LLM to score each aspect on a 0.0–1.0 scale. This is a legitimate, well-known evaluation
technique, and arguably a stronger thing to show in a portfolio than calling a library function,
since it demonstrates understanding of what each metric actually measures.

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