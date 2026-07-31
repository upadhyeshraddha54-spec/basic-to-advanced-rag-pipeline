from pathlib import Path
from datetime import datetime, timezone
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader


def _enrich_metadata(docs: List[Any], file_path: Path, file_type: str) -> List[Any]:
    """
    Attach consistent metadata to every document/chunk, regardless of loader.
    This is what makes metadata filtering and self-query retrieval possible later.
    """
    stat = file_path.stat()
    ingested_at = datetime.now(timezone.utc).isoformat()

    for doc in docs:
        # Preserve anything the loader already set (e.g. 'page' for PDFs)
        doc.metadata.update({
            "file_type": file_type,              # e.g. "pdf", "txt", "csv", "xlsx", "docx", "json"
            "file_name": file_path.name,          # e.g. "report.pdf"
            "source_path": str(file_path),        # full resolved path
            "file_size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
            "ingested_at": ingested_at,
        })
    return docs


def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from the data directory and convert to LangChain document structure.
    Supported: PDF, TXT, CSV, Excel, Word, JSON

    Every returned document has enriched metadata:
      file_type, file_name, source_path, file_size_bytes, modified_at, ingested_at
    """
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents = []

    loader_map = [
        ("pdf", "**/*.pdf", PyPDFLoader),
        ("txt", "**/*.txt", TextLoader),
        ("csv", "**/*.csv", CSVLoader),
        ("xlsx", "**/*.xlsx", UnstructuredExcelLoader),
        ("docx", "**/*.docx", Docx2txtLoader),
        ("json", "**/*.json", JSONLoader),
    ]

    for file_type, glob_pattern, loader_cls in loader_map:
        files = list(data_path.glob(glob_pattern))
        print(f"[DEBUG] Found {len(files)} {file_type.upper()} files: {[str(f) for f in files]}")

        for file_path in files:
            print(f"[DEBUG] Loading {file_type.upper()}: {file_path}")
            try:
                loader = loader_cls(str(file_path))
                loaded = loader.load()
                loaded = _enrich_metadata(loaded, file_path, file_type)
                print(f"[DEBUG] Loaded {len(loaded)} {file_type.upper()} docs from {file_path}")
                documents.extend(loaded)
            except Exception as e:
                print(f"[ERROR] Failed to load {file_type.upper()} {file_path}: {e}")

    print(f"[DEBUG] Total loaded documents: {len(documents)}")
    return documents


# Example usage
if __name__ == "__main__":
    docs = load_all_documents("data")
    print(f"Loaded {len(docs)} documents.")
    if docs:
        print("Example document metadata:", docs[0].metadata)
        print("Example document content preview:", docs[0].page_content[:200])