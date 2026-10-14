"""Pydantic schemas for Hybrid Retrieval and Reranking."""

from typing import Any, Literal
from pydantic import BaseModel, Field


class RetrievalFilter(BaseModel):
    """Filters for scoping retrieval queries."""
    tenant_id: str = "tenant_default"
    document_ids: list[str] | None = None
    file_types: list[str] | None = None
    metadata_filters: dict[str, Any] | None = None


class RetrievalQuery(BaseModel):
    """Request schema for document search / retrieval."""
    query: str = Field(..., min_length=2, description="Search query string")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of chunks to return")
    search_mode: Literal["hybrid", "vector", "keyword"] = Field(
        default="hybrid", description="Search strategy"
    )
    alpha: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Hybrid weighting (1.0 = pure vector, 0.0 = pure keyword)",
    )
    rerank: bool = Field(default=True, description="Whether to apply cross-encoder reranking")
    filters: RetrievalFilter = Field(default_factory=RetrievalFilter)


class ChunkMatch(BaseModel):
    """Single retrieved chunk match with debug scoring attributes."""
    chunk_id: str
    document_id: str
    document_title: str
    filename: str
    page_number: int | None
    chunk_index: int
    content: str
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None
    rank: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    """Debuggable response from the retrieval engine."""
    query: str
    total_matches: int
    search_mode: str
    alpha: float
    reranking_applied: bool
    latency_ms: float
    matches: list[ChunkMatch]
