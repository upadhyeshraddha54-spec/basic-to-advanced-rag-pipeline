from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The natural language question to answer")
    top_k: int = Field(5, ge=1, le=20, description="Number of final chunks to use for generation")
    metadata_filter: Optional[Dict[str, Any]] = Field(
        None, description='Optional filter, e.g. {"file_type": "pdf"}'
    )


class RetrievedChunk(BaseModel):
    file_name: Optional[str] = None
    file_type: Optional[str] = None
    text: str
    score: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    answer: str
    pipeline: str  # "baseline" or "advanced"
    contexts: List[RetrievedChunk]
    num_contexts_used: int


class HealthResponse(BaseModel):
    status: str
    baseline_ready: bool
    advanced_ready: bool