"""Hybrid Search combining pgvector dense similarity and full-text keyword search via RRF."""

from typing import Any
from src.config import settings
from src.schemas.retrieval import ChunkMatch


def apply_reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    rrf_k: int = settings.RRF_K,
    alpha: float = settings.HYBRID_ALPHA,
    top_k: int = settings.TOP_K_RETRIEVAL,
) -> list[ChunkMatch]:
    """
    Fuses dense vector results and sparse keyword results using Reciprocal Rank Fusion (RRF)
    and weighted linear score combination.
    """
    merged_items: dict[str, dict[str, Any]] = {}

    # Process dense results
    for rank, item in enumerate(dense_results, start=1):
        cid = item["chunk_id"]
        dense_rrf = 1.0 / (rrf_k + rank)
        merged_items[cid] = {
            **item,
            "dense_rank": rank,
            "sparse_rank": 9999,
            "dense_rrf": dense_rrf,
            "sparse_rrf": 0.0,
            "dense_score": item.get("dense_score", 0.0),
            "sparse_score": 0.0,
        }

    # Process sparse results
    for rank, item in enumerate(sparse_results, start=1):
        cid = item["chunk_id"]
        sparse_rrf = 1.0 / (rrf_k + rank)
        if cid in merged_items:
            merged_items[cid]["sparse_rank"] = rank
            merged_items[cid]["sparse_rrf"] = sparse_rrf
            merged_items[cid]["sparse_score"] = item.get("sparse_score", 0.0)
        else:
            merged_items[cid] = {
                **item,
                "dense_rank": 9999,
                "sparse_rank": rank,
                "dense_rrf": 0.0,
                "sparse_rrf": sparse_rrf,
                "dense_score": 0.0,
                "sparse_score": item.get("sparse_score", 0.0),
            }

    # Compute final fused score
    results: list[dict[str, Any]] = []
    for cid, data in merged_items.items():
        rrf_score = data["dense_rrf"] + data["sparse_rrf"]

        # Linear weighted score
        d_score = data["dense_score"] or 0.0
        s_score = data["sparse_score"] or 0.0
        weighted_score = (alpha * d_score) + ((1.0 - alpha) * s_score)

        # Final unified relevance score combining RRF and normalized signals
        final_score = round((rrf_score * 50.0 + weighted_score) / 2.0, 4)

        data["fused_score"] = final_score
        results.append(data)

    # Sort descending by fused score
    results.sort(key=lambda x: x["fused_score"], reverse=True)

    matches: list[ChunkMatch] = []
    for rank, item in enumerate(results[:top_k], start=1):
        matches.append(
            ChunkMatch(
                chunk_id=item["chunk_id"],
                document_id=item["document_id"],
                document_title=item["document_title"],
                filename=item["filename"],
                page_number=item["page_number"],
                chunk_index=item["chunk_index"],
                content=item["content"],
                score=item["fused_score"],
                dense_score=item.get("dense_score"),
                sparse_score=item.get("sparse_score"),
                rank=rank,
                metadata=item.get("metadata", {}),
            )
        )

    return matches
