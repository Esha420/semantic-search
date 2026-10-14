"""NVIDIA NIM LLM Provider implementation."""

import asyncio
from typing import Any
import httpx

from src.config import settings
from src.generation.base import BaseLLMProvider
from src.logger import logger


class NvidiaLLMProvider(BaseLLMProvider):
    """NVIDIA NIM Chat Completion API integration."""

    def __init__(
        self,
        api_key: str = settings.NVIDIA_API_KEY,
        base_url: str = settings.NVIDIA_BASE_URL,
        model: str = settings.LLM_MODEL,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    async def generate_response(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, dict[str, int]]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature if temperature is not None else settings.LLM_TEMPERATURE,
            "max_tokens": max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS,
        }

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=settings.LLM_REQUEST_TIMEOUT) as client:
                    response = await client.post(url, headers=headers, json=payload)

                    if response.status_code == 200:
                        data = response.json()
                        choice = data["choices"][0]
                        content = choice["message"]["content"]
                        usage = data.get("usage", {})
                        token_usage = {
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        }
                        return content, token_usage
                    elif response.status_code in (429, 500, 502, 503):
                        wait_sec = 2.0 * attempt
                        logger.warning(
                            f"NVIDIA LLM API returned {response.status_code}, retrying in {wait_sec}s..."
                        )
                        await asyncio.sleep(wait_sec)
                    else:
                        err_text = response.text
                        logger.error(f"NVIDIA LLM API error {response.status_code}: {err_text}")
                        raise RuntimeError(f"NVIDIA LLM error {response.status_code}: {err_text}")

            except httpx.RequestError as exc:
                if attempt == max_retries:
                    logger.error(f"Failed to connect to NVIDIA LLM API: {exc}")
                    raise
                await asyncio.sleep(1.5 * attempt)

        raise RuntimeError("Failed to generate response after maximum retries.")
