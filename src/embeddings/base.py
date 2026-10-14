"""Abstract Base Class for Embedding Providers."""

from abc import ABC, abstractmethod
from typing import Literal


class BaseEmbeddingProvider(ABC):
    """Protocol for dense vector embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the vector dimensions produced by the model."""
        pass

    @abstractmethod
    async def embed_texts(
        self, texts: list[str], input_type: Literal["passage", "query"] = "passage"
    ) -> list[list[float]]:
        """Embeds a batch of texts as dense vectors."""
        pass

    async def embed_query(self, query: str) -> list[float]:
        """Embeds a single query string."""
        results = await self.embed_texts([query], input_type="query")
        return results[0]
