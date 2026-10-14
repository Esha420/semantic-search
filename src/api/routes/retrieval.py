"""Hybrid and debuggable retrieval endpoints."""

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_tenant, get_db, verify_api_key
from src.retrieval.engine import RetrievalEngine
from src.schemas.retrieval import RetrievalQuery, RetrievalResponse

router = APIRouter(prefix="/retrieval", tags=["Retrieval Engine"])


@router.post(
    "/search",
    response_model=RetrievalResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)],
)
async def search_documents(
    query: RetrievalQuery,
    tenant_id: str = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
) -> RetrievalResponse:
    """
    Executes hybrid, dense vector, or keyword search across ingested document chunks.
    - Demonstrates Reciprocal Rank Fusion (RRF)
    - Returns dense score, sparse score, fused score, and reranking details
    - Scoped strictly to the requested tenant
    """
    # Force tenant isolation
    query.filters.tenant_id = tenant_id

    engine = RetrievalEngine()
    response = await engine.search(session=db, query=query)
    return response
