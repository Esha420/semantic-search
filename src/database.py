"""Database connection, engine configuration, and session management."""

from collections.abc import AsyncGenerator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from src.config import settings
from src.logger import logger


class Base(DeclarativeBase):
    """Base declarative class for all SQLAlchemy models."""
    pass


# Async engine creation
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_pre_ping=True,
)

async_session_maker = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for yielding database sessions."""
    async with async_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Initializes vector extensions and schema tables if not created."""
    try:
        async with engine.begin() as conn:
            # Enable pgvector and full-text search extensions
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))
            # Create all tables registered with Base
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database extensions and tables initialized successfully.")
    except Exception as exc:
        logger.warning(
            f"Database init warning (expected if Postgres is not reachable yet or in mock test): {exc}"
        )
