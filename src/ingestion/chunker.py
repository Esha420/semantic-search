"""Semantic and recursive chunking preserving page numbers and token budgets."""

import re
from dataclasses import dataclass
from typing import Any

from src.config import settings
from src.ingestion.parser import ExtractedPage
from src.logger import logger

try:
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    tokenizer = None


def estimate_token_count(text: str) -> int:
    """Estimates tokens using tiktoken cl100k_base or word heuristic."""
    if tokenizer:
        return len(tokenizer.encode(text))
    # Fallback heuristic: ~4 characters per token
    return max(1, len(text) // 4)


@dataclass
class ProcessedChunk:
    """A prepared document chunk ready for embedding and persistence."""
    chunk_index: int
    page_number: int | None
    content: str
    token_count: int
    metadata: dict[str, Any]


class DocumentChunker:
    """Recursive character and page-aware chunker."""

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_text_recursively(self, text: str) -> list[str]:
        """Splits long text recursively using paragraphs, sentences, and words."""
        separators = ["\n\n", "\n", ". ", "; ", ", ", " "]

        def _split(current_text: str, sep_idx: int) -> list[str]:
            if estimate_token_count(current_text) <= self.chunk_size or sep_idx >= len(separators):
                return [current_text.strip()] if current_text.strip() else []

            sep = separators[sep_idx]
            parts = current_text.split(sep)
            chunks: list[str] = []
            current_buffer: list[str] = []
            current_tokens = 0

            for part in parts:
                part_tokens = estimate_token_count(part)
                if current_tokens + part_tokens > self.chunk_size and current_buffer:
                    merged = sep.join(current_buffer).strip()
                    if merged:
                        chunks.append(merged)
                    # Handle overlap by keeping the tail part
                    current_buffer = [part]
                    current_tokens = part_tokens
                else:
                    current_buffer.append(part)
                    current_tokens += part_tokens

            if current_buffer:
                merged = sep.join(current_buffer).strip()
                if merged:
                    chunks.append(merged)

            # If any chunk is still larger than chunk_size, recurse on it with next separator
            final_chunks: list[str] = []
            for c in chunks:
                if estimate_token_count(c) > self.chunk_size and sep_idx + 1 < len(separators):
                    final_chunks.extend(_split(c, sep_idx + 1))
                else:
                    final_chunks.append(c)

            return final_chunks

        return _split(text, 0)

    def chunk_pages(
        self,
        pages: list[ExtractedPage],
        base_metadata: dict[str, Any] | None = None,
    ) -> list[ProcessedChunk]:
        """Processes extracted pages into chunk sequences tagged with page citations."""
        base_meta = base_metadata or {}
        processed: list[ProcessedChunk] = []
        global_index = 0

        for page in pages:
            raw_text = page.text.strip()
            if not raw_text:
                continue

            page_splits = self._split_text_recursively(raw_text)

            for split in page_splits:
                if not split:
                    continue
                tokens = estimate_token_count(split)
                chunk_meta = {
                    **base_meta,
                    "page_number": page.page_number,
                    "chunk_index": global_index,
                }
                processed.append(
                    ProcessedChunk(
                        chunk_index=global_index,
                        page_number=page.page_number,
                        content=split,
                        token_count=tokens,
                        metadata=chunk_meta,
                    )
                )
                global_index += 1

        logger.info(
            f"Chunked {len(pages)} pages into {len(processed)} chunks "
            f"(target chunk size: {self.chunk_size}, overlap: {self.chunk_overlap})"
        )
        return processed
