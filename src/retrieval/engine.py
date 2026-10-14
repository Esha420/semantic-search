"""Unified Retrieval Service orchestrating Vector, Full-Text, and Hybrid Search."""

import time
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.embeddings import get_embedding_provider
from src.embeddings.base import BaseEmbeddingProvider
from src.logger import logger
from src.retrieval.fts_search import search_full_text
from src.retrieval.hybrid import apply_reciprocal_rank_fusion
from src.retrieval.reranker import ContextualReranker
from src.retrieval.vector_search import search_vector_similarity
from src.schemas.retrieval import RetrievalFilter, RetrievalQuery, RetrievalResponse


class RetrievalEngine:
    """Unified engine for dense, sparse, and hybrid search with reranking."""

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider | None = None,
        reranker: ContextualReranker | None = None,
    ):
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.reranker = reranker or ContextualReranker()

    async def search(
        self,
        session: AsyncSession,
        query: RetrievalQuery,
    ) -> RetrievalResponse:
        """Executes retrieval according to requested search mode and filters."""
        start_time = time.perf_counter()

        dense_candidates = []
        sparse_candidates = []

        # 1. Vector Search
        if query.search_mode in ("hybrid", "vector"):
            query_vector = await self.embedding_provider.embed_query(query.query)
            dense_candidates = await search_vector_similarity(
                session=session,
                query_vector=query_vector,
                top_k=query.top_k * 2,
                filters=query.filters,
            )

        # 2. Full-Text Search
        if query.search_mode in ("hybrid", "keyword"):
            sparse_candidates = await search_full_text(
                session=session,
                query_text=query.query,
                top_k=query.top_k * 2,
                filters=query.filters,
            )

        # 3. Combine / Fuse results
        if query.search_mode == "vector":
            matches = [
                apply_reciprocal_rank_fusion(dense_candidates, [], top_k=query.top_k)
            ][0]
        elif query.search_mode == "keyword":
            matches = [
                apply_reciprocal_rank_fusion([], sparse_candidates, top_k=query.top_k)
            ][0]
        else:  # hybrid
            matches = apply_reciprocal_rank_fusion(
                dense_results=dense_candidates,
                sparse_results=sparse_candidates,
                rrf_k=settings.RRF_K,
                alpha=query.alpha,
                top_k=query.top_k * 2,
            )

        # 4. Contextual Reranking
        reranking_applied = False
        if query.rerank and self.reranker and matches:
            matches = self.reranker.rerank(query=query.query, matches=matches, top_k=query.top_k)
            reranking_applied = True
        else:
            matches = matches[: query.top_k]

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        return RetrievalResponse(
            query=query.query,
            total_matches=len(matches),
            search_mode=query.search_mode,
            alpha=query.alpha,
            reranking_applied=reranking_applied,
            latency_ms=elapsed_ms,
            matches=matches,
        )
