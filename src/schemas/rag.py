"""Pydantic schemas for Grounded RAG Generation, Citations, and Abstention."""

from typing import Any, Literal
from pydantic import BaseModel, Field
from src.schemas.retrieval import RetrievalFilter


class RAGQueryRequest(BaseModel):
    """Request payload for RAG-powered document question answering."""
    query: str = Field(..., min_length=2, description="User question to be answered from corpus")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context passages to retrieve")
    search_mode: Literal["hybrid", "vector", "keyword"] = "hybrid"
    rerank: bool = True
    alpha: float = Field(default=0.65, ge=0.0, le=1.0)
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=50, le=2048)
    filters: RetrievalFilter = Field(default_factory=RetrievalFilter)


class Citation(BaseModel):
    """Specific evidence citation linking the answer to a source passage."""
    citation_id: int = Field(..., description="Citation marker e.g. [1], [2]")
    document_id: str
    document_title: str
    filename: str
    page_number: int | None
    chunk_index: int
    snippet: str
    relevance_score: float


class AbstentionReason(BaseModel):
    """Structured details when the RAG model refrains from answering."""
    code: Literal[
        "NO_DOCUMENTS_FOUND",
        "LOW_RETRIEVAL_CONFIDENCE",
        "INSUFFICIENT_EVIDENCE",
        "OUT_OF_DOMAIN",
    ]
    details: str


class LatencyBreakdown(BaseModel):
    """Detailed latency profiling for each RAG pipeline phase."""
    embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


class TokenUsage(BaseModel):
    """Token consumption metadata."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class RAGResponse(BaseModel):
    """Complete, grounded RAG response with citations and traceability."""
    query: str
    answer: str
    is_abstained: bool
    abstention_reason: AbstentionReason | None = None
    citations: list[Citation] = Field(default_factory=list)
    model_name: str
    latency: LatencyBreakdown
    usage: TokenUsage
    request_id: str
    timestamp: str
