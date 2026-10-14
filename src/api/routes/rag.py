"""Grounded RAG query endpoint with source evidence citations and abstention."""

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_tenant, get_db, verify_api_key
from src.generation.engine import RAGEngine
from src.logger import request_id_ctx
from src.schemas.rag import RAGQueryRequest, RAGResponse

router = APIRouter(prefix="/rag", tags=["Grounded Generation"])


@router.post(
    "/query",
    response_model=RAGResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)],
)
async def query_rag(
    payload: RAGQueryRequest,
    tenant_id: str = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
) -> RAGResponse:
    """
    Answers a question grounded strictly in retrieved document evidence.
    - Preserves verifiable source citations [1], [2] linked to exact document & page
    - Explicitly abstains if evidence is absent or retrieval confidence is below threshold
    - Immune to prompt injection instructions embedded in documents
    - Scoped to authenticated tenant
    """
    # Enforce tenant isolation
    payload.filters.tenant_id = tenant_id

    req_id = request_id_ctx.get()
    engine = RAGEngine()
    response = await engine.answer_query(session=db, request=payload, request_id=req_id)
    return response
