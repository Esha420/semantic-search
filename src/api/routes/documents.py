"""Document ingestion and lifecycle management endpoints."""

from typing import Any
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_tenant, get_db, verify_api_key
from src.config import settings
from src.ingestion.pipeline import IngestionPipeline
from src.logger import logger
from src.models.chunk import DocumentChunk
from src.models.document import Document
from src.schemas.document import (
    DocumentChunkResponse,
    DocumentListResponse,
    DocumentResponse,
    IngestionStatusResponse,
)

router = APIRouter(prefix="/documents", tags=["Document Ingestion"])


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_api_key)],
)
async def upload_document(
    file: UploadFile = File(..., description="PDF, TXT, or Markdown document to ingest"),
    title: str | None = Form(default=None, description="Optional custom document title"),
    tenant_id: str = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """
    Ingests a document file into the RAG vector store.
    - Validates file format and size
    - Calculates SHA-256 hash for idempotent deduplication
    - Extracts text per page via PyMuPDF
    - Recursively chunks preserving page citation markers
    - Generates dense embeddings
    - Persists document and chunk records
    """
    filename = file.filename or "uploaded_document.txt"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    pipeline = IngestionPipeline()
    try:
        doc = await pipeline.ingest_file(
            session=db,
            filename=filename,
            file_bytes=file_bytes,
            tenant_id=tenant_id,
            title=title,
        )
        return DocumentResponse.model_validate(doc)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        logger.error(f"Ingestion failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal document ingestion error: {exc}",
        )


@router.get(
    "",
    response_model=DocumentListResponse,
    dependencies=[Depends(verify_api_key)],
)
async def list_documents(
    tenant_id: str = Depends(get_current_tenant),
    status_filter: str | None = Query(default=None, alias="status"),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> DocumentListResponse:
    """Retrieves paginated list of ingested documents for tenant."""
    query = select(Document).where(Document.tenant_id == tenant_id)
    count_query = select(func.count(Document.id)).where(Document.tenant_id == tenant_id)

    if status_filter:
        query = query.where(Document.status == status_filter)
        count_query = count_query.where(Document.status == status_filter)

    query = query.order_by(Document.created_at.desc()).offset(skip).limit(limit)

    total = (await db.execute(count_query)).scalar() or 0
    result = await db.execute(query)
    docs = result.scalars().all()

    return DocumentListResponse(
        total=total,
        items=[DocumentResponse.model_validate(d) for d in docs],
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    dependencies=[Depends(verify_api_key)],
)
async def get_document(
    document_id: str,
    tenant_id: str = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """Fetches document metadata by ID."""
    stmt = select(Document).where(Document.id == document_id, Document.tenant_id == tenant_id)
    res = await db.execute(stmt)
    doc = res.scalars().first()

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )
    return DocumentResponse.model_validate(doc)


@router.get(
    "/{document_id}/status",
    response_model=IngestionStatusResponse,
    dependencies=[Depends(verify_api_key)],
)
async def get_ingestion_status(
    document_id: str,
    tenant_id: str = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
) -> IngestionStatusResponse:
    """Lightweight polling endpoint to check asynchronous ingestion status."""
    stmt = select(Document).where(Document.id == document_id, Document.tenant_id == tenant_id)
    res = await db.execute(stmt)
    doc = res.scalars().first()

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )

    return IngestionStatusResponse(
        document_id=doc.id,
        status=doc.status,
        total_chunks=doc.total_chunks,
        error_message=doc.error_message,
        updated_at=doc.updated_at,
    )


@router.get(
    "/{document_id}/chunks",
    response_model=list[DocumentChunkResponse],
    dependencies=[Depends(verify_api_key)],
)
async def get_document_chunks(
    document_id: str,
    tenant_id: str = Depends(get_current_tenant),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> list[DocumentChunkResponse]:
    """Retrieves paginated chunks belonging to a document."""
    stmt = (
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id, DocumentChunk.tenant_id == tenant_id)
        .order_by(DocumentChunk.chunk_index.asc())
        .offset(skip)
        .limit(limit)
    )
    res = await db.execute(stmt)
    chunks = res.scalars().all()
    return [DocumentChunkResponse.model_validate(c) for c in chunks]


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(verify_api_key)],
)
async def delete_document(
    document_id: str,
    tenant_id: str = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Deletes a document and all associated chunks."""
    stmt = select(Document).where(Document.id == document_id, Document.tenant_id == tenant_id)
    res = await db.execute(stmt)
    doc = res.scalars().first()

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )

    await db.delete(doc)
    await db.commit()
