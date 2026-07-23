# RAG Pipeline

A Retrieval-Augmented Generation (RAG) system built with LangChain, FAISS, and Groq. Load documents from multiple formats, embed them into a vector store, and query them with a fast LLM.

## Features

- Supports PDF, TXT, CSV, Word, Excel, and JSON files
- Chunking and embedding via `sentence-transformers`
- Vector search with FAISS
- LLM responses powered by Groq (`llama-3.3-70b-versatile`)
- Jupyter notebooks for exploration

## Project Structure

```
rag/
├── app.py                  # Main entry point
├── src/
│   ├── data_loader.py      # Load documents from data/
│   ├── embedding.py        # Chunk and embed documents
│   ├── vectorstore.py      # FAISS vector store (build, save, load, query)
│   └── search.py           # RAG search + Groq LLM summarization
├── notebook/
│   ├── document.ipynb      # LangChain Document structure exploration
│   └── pdf_loader.ipynb    # PDF loading and splitting exploration
├── data/                   # Place your documents here
│   ├── pdf/
│   ├── text_files/
│   └── csv/
├── requirements.txt
└── .env                    # API keys (not committed)
```

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/your-username/rag.git
cd rag
```

**2. Create and activate a virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your Groq API key**

Create a `.env` file in the root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your key at [console.groq.com](https://console.groq.com).

**5. Add your documents**

Place files in the `data/` folder. Supported formats: `.pdf`, `.txt`, `.csv`, `.docx`, `.xlsx`, `.json`

## Usage

```bash
python app.py
```

On first run it builds the FAISS index from your documents and saves it to `faiss_store/`. Subsequent runs load the existing index directly.

To rebuild the index (e.g. after adding new documents), uncomment this line in `app.py`:
```python
store.build_from_documents(docs)
```

## How It Works

1. **Load** — documents are read from the `data/` folder
2. **Chunk** — text is split into overlapping chunks (`chunk_size=1000`, `chunk_overlap=200`)
3. **Embed** — chunks are converted to vectors using `all-MiniLM-L6-v2`
4. **Store** — vectors are saved in a FAISS index
5. **Retrieve** — at query time, the top-K most similar chunks are fetched
6. **Generate** — retrieved chunks are passed as context to Groq LLM for a final answer

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `top_k` | 3–5 | Number of chunks retrieved per query |
| `chunk_size` | 1000 | Characters per chunk |
| `chunk_overlap` | 200 | Overlap between chunks |
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `llm_model` | `llama-3.3-70b-versatile` | Groq model |

## Dependencies

- [LangChain](https://github.com/langchain-ai/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [sentence-transformers](https://www.sbert.net/)
- [Groq](https://console.groq.com)
- [ChromaDB](https://www.trychroma.com/)
# rag
