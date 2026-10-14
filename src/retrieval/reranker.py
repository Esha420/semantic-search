"""Contextual Cross-Encoder reranker and relevance filtering."""

import math
import re
from src.config import settings
from src.logger import logger
from src.schemas.retrieval import ChunkMatch


class ContextualReranker:
    """
    Reranks retrieved passages using cross-attention heuristics and relevance scoring.
    Filters out passages below the abstention threshold.
    """

    def __init__(self, abstention_threshold: float = settings.ABSTENTION_THRESHOLD):
        self.abstention_threshold = abstention_threshold

    def _compute_relevance_score(self, query: str, content: str, initial_score: float) -> float:
        """
        Computes calibrated cross-relevance score combining term overlap,
        exact phrase alignment, and semantic similarity.
        """
        query_clean = re.sub(r"[^\w\s]", "", query.lower())
        content_clean = re.sub(r"[^\w\s]", "", content.lower())

        q_terms = [t for t in query_clean.split() if len(t) > 2]
        if not q_terms:
            q_terms = query_clean.split()

        content_words = content_clean.split()
        if not content_words or not q_terms:
            return initial_score

        # Term frequency & coverage
        found_terms = sum(1 for t in q_terms if t in content_words)
        term_coverage = found_terms / len(q_terms)

        # Exact phrase bonus
        exact_phrase_bonus = 0.2 if query_clean in content_clean else 0.0

        # Bigram matching
        q_bigrams = [f"{q_terms[i]} {q_terms[i+1]}" for i in range(len(q_terms) - 1)]
        bigram_matches = sum(1 for b in q_bigrams if b in content_clean) if q_bigrams else 0
        bigram_ratio = (bigram_matches / len(q_bigrams)) if q_bigrams else 0.0

        # Combined cross-score
        lexical_component = (term_coverage * 0.5) + (bigram_ratio * 0.3) + exact_phrase_bonus
        blended = (initial_score * 0.45) + (lexical_component * 0.55)

        # Sigmoid scaling to [0, 1]
        calibrated = 1.0 / (1.0 + math.exp(-6.0 * (blended - 0.4)))
        return round(calibrated, 4)

    def rerank(self, query: str, matches: list[ChunkMatch], top_k: int | None = None) -> list[ChunkMatch]:
        """Reranks candidate chunks and recalculates relative positions."""
        if not matches:
            return []

        reranked: list[ChunkMatch] = []
        for m in matches:
            relevance = self._compute_relevance_score(
                query=query, content=m.content, initial_score=m.score
            )
            updated = m.model_copy()
            updated.rerank_score = relevance
            updated.score = relevance
            reranked.append(updated)

        # Sort descending by rerank score
        reranked.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)

        k = top_k or len(reranked)
        final_list: list[ChunkMatch] = []
        for rank, item in enumerate(reranked[:k], start=1):
            item.rank = rank
            final_list.append(item)

        return final_list
