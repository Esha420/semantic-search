"""FastAPI Dependency Injection providers."""

from collections.abc import AsyncGenerator
from fastapi import Header, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.database import get_db_session

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Provides async database session."""
    async for session in get_db_session():
        yield session


async def verify_api_key(
    api_key: str | None = Security(api_key_header),
) -> str | None:
    """Enforces API Key authentication if enabled in settings."""
    if not settings.API_KEY_AUTH_ENABLED:
        return None

    if not api_key or api_key != settings.APP_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )
    return api_key


async def get_current_tenant(
    x_tenant_id: str | None = Header(default=None),
) -> str:
    """Resolves tenant ID from request header or uses default."""
    return x_tenant_id or settings.DEFAULT_TENANT_ID
