"""Health check endpoints for container readiness and dependency probing."""

from typing import Any
import httpx
from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db
from src.config import settings

router = APIRouter(prefix="/health", tags=["Health & Diagnostics"])


@router.get("", status_code=status.HTTP_200_OK)
async def health_check() -> dict[str, str]:
    """Basic service health check."""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "environment": settings.ENVIRONMENT,
    }


@router.get("/live", status_code=status.HTTP_200_OK)
async def liveness_probe() -> dict[str, str]:
    """Kubernetes / Docker liveness probe."""
    return {"status": "alive"}


@router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_probe(db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """
    Deep readiness check validating:
    - PostgreSQL connection and pgvector extension
    - LLM provider connectivity
    """
    checks = {
        "database": "unknown",
        "pgvector_extension": "unknown",
        "llm_provider": "unknown",
    }
    is_ready = True

    # 1. Check PostgreSQL
    try:
        db_res = await db.execute(text("SELECT 1;"))
        if db_res.scalar() == 1:
            checks["database"] = "connected"
    except Exception as exc:
        checks["database"] = f"error: {exc}"
        is_ready = False

    # 2. Check pgvector extension
    try:
        ext_res = await db.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        )
        if ext_res.scalar() == "vector":
            checks["pgvector_extension"] = "installed"
        else:
            checks["pgvector_extension"] = "missing"
    except Exception as exc:
        checks["pgvector_extension"] = f"error: {exc}"

    # 3. Check NVIDIA NIM or LLM provider
    if settings.LLM_PROVIDER == "nvidia" and settings.NVIDIA_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                res = await client.get(
                    f"{settings.NVIDIA_BASE_URL}/models",
                    headers={"Authorization": f"Bearer {settings.NVIDIA_API_KEY}"},
                )
                checks["llm_provider"] = "reachable" if res.status_code == 200 else f"status {res.status_code}"
        except Exception as exc:
            checks["llm_provider"] = f"unreachable: {exc}"
    else:
        checks["llm_provider"] = f"active ({settings.LLM_PROVIDER})"

    return {
        "ready": is_ready,
        "checks": checks,
    }
