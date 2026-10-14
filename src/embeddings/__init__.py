"""Embedding provider exports and factory function."""

from src.config import settings
from src.embeddings.base import BaseEmbeddingProvider
from src.embeddings.mock import MockEmbeddingProvider
from src.embeddings.nvidia import NvidiaEmbeddingProvider


def get_embedding_provider(
    provider_name: str | None = None,
) -> BaseEmbeddingProvider:
    """Factory returning the configured embedding provider."""
    provider = (provider_name or settings.EMBEDDING_PROVIDER).lower()

    if provider == "nvidia":
        return NvidiaEmbeddingProvider()
    elif provider == "mock":
        return MockEmbeddingProvider()
    else:
        # Default fallback to mock if unrecognized
        return MockEmbeddingProvider()


__all__ = [
    "BaseEmbeddingProvider",
    "NvidiaEmbeddingProvider",
    "MockEmbeddingProvider",
    "get_embedding_provider",
]
