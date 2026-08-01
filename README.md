# RAG Pipeline: From Basic to Advanced

This project started as a simple Retrieval-Augmented Generation (RAG) system — load some documents,
embed them, search with FAISS, ask an LLM to answer. Over time I extended it into a full advanced RAG
pipeline, one technique at a time, wrapped it in a FastAPI service, containerized it with Docker, and
finally evaluated whether all that extra complexity actually helped.

Short answer: mostly yes, in the ways that matter, but not without a lot of real debugging along the
way. This README walks through the whole thing honestly — what was built, what broke, and what I
learned from the failures, including the Docker deployment itself.

---

## Tech Stack

- **LangChain** for document loading and text splitting
- **FAISS** for vector search
- **Groq** (`llama-3.3-70b-versatile` for generation, `llama-3.1-8b-instant` for cheaper high-volume calls)
- **sentence-transformers** (`all-MiniLM-L6-v2`) for embeddings
- **rank-bm25** for keyword search
- **cross-encoder/ms-marco-MiniLM-L-6-v2** for re-ranking
- **FastAPI + Uvicorn** for the API layer
- **Docker** for containerized deployment

---

## Part 1: The Basic Pipeline

Before any of the advanced work, this was the whole system:

1. Load documents from `data/` — PDF, TXT, CSV, DOCX, XLSX, JSON
2. Split them into overlapping chunks (1000 characters, 200 overlap)
3. Embed each chunk with `all-MiniLM-L6-v2`
4. Store the vectors in FAISS
5. At query time, pull the top-k most similar chunks
6. Hand those chunks to Groq's `llama-3.3-70b-versatile` and ask it to answer

It works, and for a lot of use cases it's genuinely fine. But it has clear gaps: only one way to
search, no filtering by document type or date, no handling for differently-worded questions, and no
way to actually measure whether it's any good. Everything below targets one of those gaps.

---

## Part 2: What Got Added, and Why

### Metadata Filtering

Every chunk gets tagged at load time with `file_type`, `file_name`, `source_path`, `file_size_bytes`,
`modified_at`, and `ingested_at`. The vector store's search function accepts a filter dict, so you can
say "only search CSV files" and it'll respect that. FAISS doesn't filter natively, so this works by
over-fetching a larger candidate pool, filtering by metadata, then trimming to the requested count.
Tested with CSV-only and PDF-only filters — worked cleanly both times.

### Hybrid Search (BM25 + FAISS)

Vector search matches meaning but can miss exact keywords or IDs. BM25 (keyword search) is the
opposite. Both run side by side, and results are merged using **Reciprocal Rank Fusion** — a chunk
that ranks well in either list gets boosted, using rank position rather than raw scores (which aren't
on comparable scales anyway). Confirmed working: for "attention mechanism," hybrid search pulled up a
relevant CSV row that pure FAISS ranked lower, because BM25 caught a keyword match the embedding
missed.

### Cross-Encoder Re-ranking

Hybrid search scores the query and each chunk separately, then compares - fast but imprecise. A
cross-encoder looks at the query and chunk *together* for a much more accurate score, applied only to
hybrid search's top 20-30 candidates since it's too slow to run on everything. Caught something real:
a few CSV rows matched on keyword overlap but turned out to be generic templated filler, not real
explanations. The reranker correctly pushed those down.

### Query Transformation

Three related techniques in one file:
- **Query Expansion** - the LLM rewrites the question a few different ways to catch different phrasing.
- **Multi-Query Retrieval** - search runs once per rewritten version, results get merged.
- **Self-Query Retriever** - the LLM splits a question like "find CSV rows about deep learning" into
  a search term plus a metadata filter automatically.

All three worked as intended. Self-query was the cleanest result - parsed correctly, every result
genuinely matched the requested filter.

### Contextual Compression

Trims each retrieved chunk down to just the relevant sentences, or drops it entirely if nothing in it
is relevant. This is the technique that caused the most real trouble:

An early version of the compression prompt was too strict, requiring close-to-literal wording overlap
with the query. On one test question, this threw away the one chunk that actually answered it, because
that chunk never used the query's exact phrasing. The LLM then answered "I don't know," tanking that
question's score to zero. Fixed it two ways: loosened the compression prompt to keep supporting
context, and added a fallback - if compression leaves too few chunks, fall back to the uncompressed
reranked chunks instead of risking a starved answer. It also turned out the test question itself used
a term the source document never used, so I reworded it too.

**The real lesson:** a technique working exactly as designed can still hurt results if it's tuned too
aggressively, or if your test data doesn't match your test questions.

### Parent Document Retriever

Small chunks search precisely but lack context; big chunks carry context but their embeddings get
diluted. This splits documents into large "parent" chunks (~2000 chars) and small "child" chunks
(~400 chars). Only children get embedded and searched; when one matches, its parent is returned for
full context. Found a real limitation: the best-matching child chunk was literally the heading
"Multi-Head Attention," but it sat right at a parent-chunk boundary, so the parent returned was
actually the adjacent section. A known tradeoff of splitting by character count instead of document
structure.

### Evaluation

Built a 7-question test set with real ground-truth answers, then measured Context Precision, Context
Recall, Faithfulness, and Answer Relevancy for the basic pipeline vs. the full advanced one.

Hit a real snag: the `ragas` library (v0.4.3) has a broken import - it pulls in `ChatVertexAI` from a
`langchain_community` path that's been removed. After failed attempts to fix it via dependency pins,
I wrote a custom LLM-as-judge evaluator instead (`eval/custom_metrics.py`), scoring each metric by
directly prompting an LLM. A legitimate, well-known technique, and arguably better to show than
calling a library function, since it means actually understanding what's being measured.

**Final numbers:**

| Metric | Basic Pipeline | Advanced Pipeline | Change |
|---|---|---|---|
| Context Precision | 0.574 | **0.679** | **+0.105 - meaningfully better** |
| Context Recall | 0.810 | 0.737 | -0.073 |
| Faithfulness | 0.896 | 0.894 | basically unchanged |
| Answer Relevancy | 0.936 | 0.921 | basically unchanged |

The advanced pipeline retrieves noticeably cleaner context without giving up faithfulness or
relevancy. The recall dip is an expected tradeoff - narrowing context down occasionally trims
something that would've added completeness, even without hurting the actual answer.

---

## Part 3: Wrapping It in an API

Once the pipelines worked, I wrapped them in a FastAPI service so they're usable outside a Python
script.

### API Architecture

Two endpoints, both backed by the same underlying pipelines:

- `POST /v1/search/baseline` - plain FAISS similarity search -> generate
- `POST /v1/search/advanced` - hybrid search -> cross-encoder rerank -> contextual compression -> generate
  (with the same compression fallback safety net used in evaluation)
- `GET /health` - reports whether both pipelines are loaded and ready

Models and indexes load **once** at startup via a FastAPI `lifespan` context manager, not per-request
- reloading a sentence-transformer or cross-encoder on every call would be far too slow. Since FAISS,
sentence-transformers, and the Groq client are all blocking/synchronous, blocking calls are offloaded
to a thread pool (`run_in_threadpool`) so the async event loop stays responsive under concurrent load.

Interactive API docs are auto-generated by FastAPI at `/docs` once the server is running.

### Production Deployment (Docker)

The app is fully containerized so it runs identically on any machine, not just the one it was
developed on.

```bash
docker compose up --build
```

This builds a multi-stage image (build dependencies in one stage, a slim runtime image in the second)
and starts the API on `http://localhost:8000`.

**A few real things that went wrong getting here, worth knowing about:**

- **CUDA bloat:** the first build attempt pulled multi-GB CUDA/cuDNN packages for PyTorch, even though
  this container only ever does CPU inference and has no GPU access. Fixed by explicitly installing
  the CPU-only PyTorch build (`--index-url https://download.pytorch.org/whl/cpu`) before installing
  anything else.
- **Host disk space:** Docker builds failed with cryptic `input/output error` messages that turned out
  to be the host machine's disk being essentially full, not a Docker or code problem. Freeing disk
  space on the host fixed it.
- **Corrupted Docker state:** after the disk-full failures, Docker's own internal metadata got
  corrupted, requiring a full `docker system prune` and Docker Desktop restart before builds could
  succeed again.
- **Missing dependency:** `faiss-cpu` was installed locally but never actually listed in
  `requirements.txt`, so the container crashed with `ModuleNotFoundError: No module named 'faiss'`
  until it was added explicitly, both to `requirements.txt` and the Dockerfile.
- **Build cache:** added `--mount=type=cache` on pip install steps so that after the first (necessarily
  slow) build, any future dependency fix only takes about a minute instead of triggering a full
  re-download.

None of these were exotic problems - they're exactly the kind of environment/dependency friction that
comes up in real deployment work, and worth being upfront about rather than pretending the first
`docker build` just worked.

---

## Project Structure

```
rag/
|-- app/
|   |-- main.py                # FastAPI gateway (baseline + advanced endpoints)
|   `-- schemas.py              # Pydantic request/response models
|-- src/
|   |-- data_loader.py          # loads files, tags metadata
|   |-- vectorstore.py          # FAISS store + metadata filtering
|   |-- hybrid_search.py        # BM25 + FAISS with RRF
|   |-- reranker.py             # cross-encoder re-ranking
|   |-- query_transform.py      # expansion, multi-query, self-query
|   |-- compressor.py           # contextual compression
|   |-- parent_retriever.py     # parent/child chunking
|   |-- embedding.py            # chunking + embedding
|   `-- search.py               # original basic pipeline
|-- eval/
|   |-- qa_testset.json         # 7 test questions with ground truth
|   |-- custom_metrics.py       # LLM-as-judge metric implementations
|   |-- run_ragas.py            # runs basic vs advanced, evaluates both
|   |-- results_baseline.csv
|   `-- results_advanced.csv
|-- data/
|   |-- pdf/
|   |-- text_files/
|   `-- csv/
|-- faiss_store/                # main index (mounted as a volume in Docker)
|-- faiss_store_parent/         # parent-retriever's separate index
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
`-- .env                        # GROQ_API_KEY, not committed
```

---

## Setup (Running Locally, Without Docker)

```bash
git clone https://github.com/upadhyeshraddha54-spec/rag.git
cd rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

Build the indexes (only needed once, or after adding new data):
```bash
python -m src.hybrid_search
```

Run the API:
```bash
uvicorn app.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` to try it interactively, or run individual pipeline modules
directly (always as a module, from the repo root):
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

## Setup (Running with Docker)

```bash
docker compose up --build
```

Then, in another terminal:
```bash
curl http://localhost:8000/health
```

---

## What I'd Tell Someone Else Doing This

- Adding more techniques doesn't automatically improve results - each one needs tuning, and
  compression especially can hurt more than it helps if it's too aggressive.
- Self-query retrieval is only as good as your metadata.
- Check your test data actually contains what your test questions ask about - a corpus/question
  mismatch can look exactly like a pipeline bug.
- When a library breaks (looking at you, `ragas`), building a small custom version can genuinely be
  the better choice, since it forces you to understand what you're measuring.
- Budget your API tokens - evaluation runs far more LLM calls than you'd expect.
- Docker failures are very often not about your code - disk space and corrupted local state caused
  more of the real debugging here than anything in the Dockerfile itself.

---

## Pushing to GitHub

If the repo isn't connected yet:
```bash
git init
git add .
git commit -m "Advanced RAG: hybrid search, reranking, query transformation, compression, parent retrieval, evaluation, FastAPI, Docker"
git branch -M main
git remote add origin https://github.com/upadhyeshraddha54-spec/rag.git
git push -u origin main
```

If it's already connected:
```bash
git add .
git commit -m "Add FastAPI gateway and Docker deployment"
git push
```

Before committing, make sure your `.gitignore` includes:
```
.env
.venv/
__pycache__/
faiss_store/
faiss_store_parent/
*.pkl
```

Never commit `.env`. If it's ever committed by accident, rotate your Groq API key immediately, not
just remove the file.