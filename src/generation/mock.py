"""Deterministic mock LLM provider for unit tests and local simulation."""

from src.generation.base import BaseLLMProvider


class MockLLMProvider(BaseLLMProvider):
    """Simulates LLM responses with citations or abstention based on context presence."""

    def __init__(self, model_name: str = "mock-llm-model"):
        self._model = model_name

    @property
    def model_name(self) -> str:
        return self._model

    async def generate_response(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, dict[str, int]]:
        user_message = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_message = m.get("content", "")
                break

        # Check if the context contains passages
        if "[Passage 1]" in user_message and "unsupported_question" not in user_message.lower():
            # Extract first sentence from passage 1
            lines = user_message.split("[Passage 1]")[-1].splitlines()
            first_sentence = "According to the documented policies"
            for line in lines:
                if line.strip() and not line.strip().startswith("("):
                    first_sentence = line.strip()[:120]
                    break
            answer = f"Based on the provided documentation, {first_sentence} [1]."
        else:
            answer = "INSUFFICIENT_EVIDENCE: The provided documents do not contain sufficient evidence to answer this question."

        usage = {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160}
        return answer, usage
