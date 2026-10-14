"""Application configuration managed via Pydantic Settings."""

from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for Document Intelligence RAG API."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Core Application Settings
    ENVIRONMENT: Literal["development", "staging", "production", "test"] = "development"
    APP_NAME: str = "Document Intelligence RAG API"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # Database Settings (PostgreSQL + pgvector)
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "document_intelligence"
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/document_intelligence"
    SYNC_DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/document_intelligence"
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10

    # Redis & Celery
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"
    USE_CELERY: bool = False

    # LLM Settings
    LLM_PROVIDER: Literal["nvidia", "openai", "gemini", "mock"] = "nvidia"
    NVIDIA_API_KEY: str = ""
    NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    LLM_MODEL: str = "meta/llama-3.2-11b-vision-instruct"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1024
    LLM_REQUEST_TIMEOUT: float = 45.0

    # Embedding Settings
    EMBEDDING_PROVIDER: Literal["nvidia", "local", "mock"] = "nvidia"
    EMBEDDING_MODEL: str = "nvidia/nemotron-3-embed-1b"
    EMBEDDING_DIMENSIONS: int = 2048
    EMBEDDING_BATCH_SIZE: int = 32

    # Ingestion & Chunking Parameters
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 80
    MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB
    ALLOWED_EXTENSIONS: list[str] = Field(default_factory=lambda: [".pdf", ".txt", ".md"])

    # Retrieval & Ranking Parameters
    TOP_K_RETRIEVAL: int = 5
    HYBRID_ALPHA: float = 0.65  # Weight for vector similarity (1 - alpha for text search)
    RRF_K: int = 60  # Constant for Reciprocal Rank Fusion
    RERANKING_ENABLED: bool = True
    ABSTENTION_THRESHOLD: float = 0.35  # Scores below this trigger abstention / warning

    # Security & Access Control
    API_KEY_AUTH_ENABLED: bool = False
    APP_API_KEY: str = "verisk-doc-intel-secret-key-2026"
    DEFAULT_TENANT_ID: str = "tenant_default"


settings = Settings()
