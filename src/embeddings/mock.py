"""Deterministic mock embedding provider for testing and offline runs."""

import hashlib
import math
from typing import Literal

from src.config import settings
from src.embeddings.base import BaseEmbeddingProvider


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Generates deterministic unit-normalized embeddings based on text hashes."""

    def __init__(self, dimension: int = settings.EMBEDDING_DIMENSIONS):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _generate_vector(self, text: str) -> list[float]:
        # Hash text to generate pseudo-random deterministic floats
        hasher = hashlib.sha256(text.encode("utf-8"))
        seed = int.from_bytes(hasher.digest()[:8], byteorder="big")

        raw_vector: list[float] = []
        for i in range(self._dimension):
            # Simple LCG pseudorandom sequence
            seed = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            val = ((seed >> 32) / 0xFFFFFFFF) * 2.0 - 1.0
            raw_vector.append(val)

        # L2 Normalize
        norm = math.sqrt(sum(x * x for x in raw_vector)) or 1.0
        return [x / norm for x in raw_vector]

    async def embed_texts(
        self, texts: list[str], input_type: Literal["passage", "query"] = "passage"
    ) -> list[list[float]]:
        return [self._generate_vector(t) for t in texts]
