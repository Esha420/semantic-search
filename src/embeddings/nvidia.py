"""NVIDIA NIM Embedding Provider implementation."""

import asyncio
from typing import Literal
import httpx

from src.config import settings
from src.embeddings.base import BaseEmbeddingProvider
from src.logger import logger


class NvidiaEmbeddingProvider(BaseEmbeddingProvider):
    """NVIDIA NIM API integration for dense embeddings."""

    def __init__(
        self,
        api_key: str = settings.NVIDIA_API_KEY,
        base_url: str = settings.NVIDIA_BASE_URL,
        model: str = settings.EMBEDDING_MODEL,
        dimension: int = settings.EMBEDDING_DIMENSIONS,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def _call_api_with_retry(
        self, client: httpx.AsyncClient, batch: list[str], input_type: str, max_retries: int = 3
    ) -> list[list[float]]:
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "model": self.model,
            "input": batch,
            "input_type": input_type,
        }

        for attempt in range(1, max_retries + 1):
            try:
                response = await client.post(url, headers=headers, json=payload, timeout=30.0)
                if response.status_code == 200:
                    data = response.json()
                    # Preserve order by data[i].index or array order
                    items = data.get("data", [])
                    return [item["embedding"] for item in items]
                elif response.status_code in (429, 500, 502, 503):
                    wait_time = 1.5 * attempt
                    logger.warning(
                        f"NVIDIA embedding API returned {response.status_code}, retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    error_text = response.text
                    logger.error(f"NVIDIA embedding API error {response.status_code}: {error_text}")
                    raise RuntimeError(f"NVIDIA API Error {response.status_code}: {error_text}")
            except httpx.RequestError as exc:
                if attempt == max_retries:
                    logger.error(f"Failed to connect to NVIDIA embedding API after {max_retries} attempts: {exc}")
                    raise
                await asyncio.sleep(1.0 * attempt)

        raise RuntimeError("Exceeded maximum retries for NVIDIA embedding generation.")

    async def embed_texts(
        self, texts: list[str], input_type: Literal["passage", "query"] = "passage"
    ) -> list[list[float]]:
        """Embeds list of texts in chunks according to EMBEDDING_BATCH_SIZE."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        batch_size = settings.EMBEDDING_BATCH_SIZE

        async with httpx.AsyncClient() as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch_embeddings = await self._call_api_with_retry(client, batch, input_type)
                all_embeddings.extend(batch_embeddings)

        return all_embeddings
