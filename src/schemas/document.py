"""Pydantic schemas for Document Ingestion and status."""

from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class DocumentCreate(BaseModel):
    """Schema for document upload options."""
    title: str | None = None
    tenant_id: str = "tenant_default"
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class DocumentChunkResponse(BaseModel):
    """Response schema for an individual document chunk."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    chunk_index: int
    page_number: int | None
    content: str
    token_count: int
    metadata_json: dict[str, Any]
    created_at: datetime


class DocumentResponse(BaseModel):
    """Full document metadata response schema."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    tenant_id: str
    title: str
    filename: str
    file_type: str
    file_size_bytes: int
    content_hash: str
    version: int
    status: str
    error_message: str | None
    total_chunks: int
    metadata_json: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    """Paginated list of ingested documents."""
    total: int
    items: list[DocumentResponse]


class IngestionStatusResponse(BaseModel):
    """Lightweight polling status response for background ingestion."""
    document_id: str
    status: str  # pending, processing, completed, failed
    total_chunks: int
    error_message: str | None
    updated_at: datetime
