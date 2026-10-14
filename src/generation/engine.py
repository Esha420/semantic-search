"""Grounded RAG generation engine with evidence citation parsing and abstention."""

import re
import time
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.generation.base import BaseLLMProvider
from src.generation.mock import MockLLMProvider
from src.generation.nvidia import NvidiaLLMProvider
from src.generation.prompt_templates import build_rag_messages
from src.logger import logger
from src.retrieval.engine import RetrievalEngine
from src.schemas.rag import (
    AbstentionReason,
    Citation,
    LatencyBreakdown,
    RAGQueryRequest,
    RAGResponse,
    TokenUsage,
)
from src.schemas.retrieval import RetrievalQuery


def get_llm_provider(provider_type: str | None = None) -> BaseLLMProvider:
    """Factory returning configured LLM provider."""
    provider = (provider_type or settings.LLM_PROVIDER).lower()
    if provider == "nvidia":
        return NvidiaLLMProvider()
    elif provider == "mock":
        return MockLLMProvider()
    else:
        return MockLLMProvider()


class RAGEngine:
    """Full RAG Orchestrator: retrieval, context assembly, LLM prompting, and verification."""

    def __init__(
        self,
        retrieval_engine: RetrievalEngine | None = None,
        llm_provider: BaseLLMProvider | None = None,
    ):
        self.retrieval_engine = retrieval_engine or RetrievalEngine()
        self.llm_provider = llm_provider or get_llm_provider()

    async def answer_query(
        self,
        session: AsyncSession,
        request: RAGQueryRequest,
        request_id: str = "system",
    ) -> RAGResponse:
        """
        Executes end-to-end grounded Q&A with source citations and abstention check.
        """
        overall_start = time.perf_counter()

        # Step 1: Retrieve relevant context chunks
        retrieval_start = time.perf_counter()
        retrieval_req = RetrievalQuery(
            query=request.query,
            top_k=request.top_k,
            search_mode=request.search_mode,
            alpha=request.alpha,
            rerank=request.rerank,
            filters=request.filters,
        )
        retrieval_res = await self.retrieval_engine.search(session=session, query=retrieval_req)
        retrieval_ms = round((time.perf_counter() - retrieval_start) * 1000.0, 2)

        chunks = retrieval_res.matches

        # Case A: No documents or chunks found at all
        if not chunks:
            total_ms = round((time.perf_counter() - overall_start) * 1000.0, 2)
            return RAGResponse(
                query=request.query,
                answer="No relevant documents were found in the knowledge base to address this query.",
                is_abstained=True,
                abstention_reason=AbstentionReason(
                    code="NO_DOCUMENTS_FOUND",
                    details="No indexed chunks matched the query filters or keyword/semantic criteria.",
                ),
                citations=[],
                model_name=self.llm_provider.model_name,
                latency=LatencyBreakdown(
                    retrieval_ms=retrieval_ms,
                    generation_ms=0.0,
                    total_ms=total_ms,
                ),
                usage=TokenUsage(),
                request_id=request_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        # Case B: Low retrieval confidence check (abstention threshold)
        top_score = chunks[0].score if chunks else 0.0
        if top_score < settings.ABSTENTION_THRESHOLD:
            total_ms = round((time.perf_counter() - overall_start) * 1000.0, 2)
            return RAGResponse(
                query=request.query,
                answer=(
                    "The available documents have very low relevance to this question. "
                    "Abstaining from generating an ungrounded response."
                ),
                is_abstained=True,
                abstention_reason=AbstentionReason(
                    code="LOW_RETRIEVAL_CONFIDENCE",
                    details=(
                        f"Top retrieval score ({top_score:.2f}) is below the confidence "
                        f"abstention threshold ({settings.ABSTENTION_THRESHOLD:.2f})."
                    ),
                ),
                citations=[],
                model_name=self.llm_provider.model_name,
                latency=LatencyBreakdown(
                    retrieval_ms=retrieval_ms,
                    generation_ms=0.0,
                    total_ms=total_ms,
                ),
                usage=TokenUsage(),
                request_id=request_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        # Step 2: Build grounded prompt
        messages = build_rag_messages(query=request.query, chunks=chunks)

        # Step 3: LLM generation
        generation_start = time.perf_counter()
        raw_answer, token_usage = await self.llm_provider.generate_response(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        generation_ms = round((time.perf_counter() - generation_start) * 1000.0, 2)
        total_ms = round((time.perf_counter() - overall_start) * 1000.0, 2)

        # Step 4: Parse Abstention and Citations
        is_abstained = False
        abstention_reason = None
        citations: list[Citation] = []

        if "INSUFFICIENT_EVIDENCE" in raw_answer:
            is_abstained = True
            abstention_reason = AbstentionReason(
                code="INSUFFICIENT_EVIDENCE",
                details="The model determined that the provided passages do not contain facts to answer the question.",
            )
        else:
            # Extract citation indices from text e.g. [1], [2]
            citation_refs = re.findall(r"\[(\d+)\]", raw_answer)
            unique_indices = sorted(list(set(int(ref) for ref in citation_refs)))

            for ref_idx in unique_indices:
                if 1 <= ref_idx <= len(chunks):
                    matched_chunk = chunks[ref_idx - 1]
                    snippet = (
                        matched_chunk.content[:160] + "..."
                        if len(matched_chunk.content) > 160
                        else matched_chunk.content
                    )
                    citations.append(
                        Citation(
                            citation_id=ref_idx,
                            document_id=matched_chunk.document_id,
                            document_title=matched_chunk.document_title,
                            filename=matched_chunk.filename,
                            page_number=matched_chunk.page_number,
                            chunk_index=matched_chunk.chunk_index,
                            snippet=snippet,
                            relevance_score=matched_chunk.score,
                        )
                    )

        return RAGResponse(
            query=request.query,
            answer=raw_answer.strip(),
            is_abstained=is_abstained,
            abstention_reason=abstention_reason,
            citations=citations,
            model_name=self.llm_provider.model_name,
            latency=LatencyBreakdown(
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                total_ms=total_ms,
            ),
            usage=TokenUsage(**token_usage),
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
