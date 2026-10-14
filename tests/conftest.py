"""Pytest fixtures and configuration for Document Intelligence RAG API tests."""

from collections.abc import AsyncGenerator
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.api.app import app
from src.api.deps import get_db
from src.config import settings
from src.database import Base
from src.embeddings.mock import MockEmbeddingProvider
from src.generation.mock import MockLLMProvider
from src.generation.engine import RAGEngine
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.engine import RetrievalEngine

# Use in-memory SQLite for high-speed local unit and integration tests
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(
    TEST_DB_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)

TestSessionLocal = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provides a clean in-memory database session for each test function."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with TestSessionLocal() as session:
        yield session

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture(scope="function")
async def async_client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Provides an async HTTP test client with database override."""
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture
def mock_embedding_provider() -> MockEmbeddingProvider:
    """Deterministic mock embedding provider."""
    return MockEmbeddingProvider(dimension=settings.EMBEDDING_DIMENSIONS)


@pytest.fixture
def mock_llm_provider() -> MockLLMProvider:
    """Deterministic mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def ingestion_pipeline(mock_embedding_provider) -> IngestionPipeline:
    """Ingestion pipeline wired with mock embeddings."""
    return IngestionPipeline(embedding_provider=mock_embedding_provider)


@pytest.fixture
def rag_engine(mock_embedding_provider, mock_llm_provider) -> RAGEngine:
    """RAG engine wired with mock embeddings and mock LLM."""
    retrieval_engine = RetrievalEngine(embedding_provider=mock_embedding_provider)
    return RAGEngine(retrieval_engine=retrieval_engine, llm_provider=mock_llm_provider)
