"""Abstract Base Class for LLM Providers."""

from abc import ABC, abstractmethod
from typing import Any


class BaseLLMProvider(ABC):
    """Protocol for LLM generation adapters."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the underlying LLM."""
        pass

    @abstractmethod
    async def generate_response(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, dict[str, int]]:
        """
        Generates completion from chat messages.
        Returns (response_text, token_usage_dict).
        """
        pass
