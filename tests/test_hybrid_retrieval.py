"""Tests for Hybrid Search, Reciprocal Rank Fusion (RRF), and Contextual Reranker."""

from src.retrieval.hybrid import apply_reciprocal_rank_fusion
from src.retrieval.reranker import ContextualReranker
from src.schemas.retrieval import ChunkMatch


def test_reciprocal_rank_fusion():
    dense_results = [
        {
            "chunk_id": "c1",
            "document_id": "d1",
            "document_title": "Policy",
            "filename": "policy.txt",
            "page_number": 1,
            "chunk_index": 0,
            "content": "Fire and lightning coverage limits",
            "dense_score": 0.92,
        },
        {
            "chunk_id": "c2",
            "document_id": "d1",
            "document_title": "Policy",
            "filename": "policy.txt",
            "page_number": 2,
            "chunk_index": 1,
            "content": "Water damage deductible is $15,000",
            "dense_score": 0.81,
        },
    ]

    sparse_results = [
        {
            "chunk_id": "c2",  # Present in both dense and sparse!
            "document_id": "d1",
            "document_title": "Policy",
            "filename": "policy.txt",
            "page_number": 2,
            "chunk_index": 1,
            "content": "Water damage deductible is $15,000",
            "sparse_score": 0.95,
        },
        {
            "chunk_id": "c3",
            "document_id": "d2",
            "document_title": "Underwriting",
            "filename": "underwriting.txt",
            "page_number": 1,
            "chunk_index": 0,
            "content": "Roof age limits and depreciation",
            "sparse_score": 0.70,
        },
    ]

    # Chunk c2 appears in BOTH lists, so RRF should boost it to top position!
    fused = apply_reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        rrf_k=60,
        alpha=0.65,
        top_k=3,
    )

    assert len(fused) == 3
    # c2 has high dense + high sparse, so it ranks first
    assert fused[0].chunk_id == "c2"
    assert fused[0].score > fused[1].score


def test_contextual_reranker():
    reranker = ContextualReranker()
    query = "commercial water damage deductible"

    matches = [
        ChunkMatch(
            chunk_id="c1",
            document_id="d1",
            document_title="Policy",
            filename="policy.txt",
            page_number=1,
            chunk_index=0,
            content="General overview of property casualty lines.",
            score=0.75,
            rank=1,
        ),
        ChunkMatch(
            chunk_id="c2",
            document_id="d1",
            document_title="Policy",
            filename="policy.txt",
            page_number=2,
            chunk_index=1,
            content="Water damage deductible is $15,000 per occurrence for plumbing.",
            score=0.70,
            rank=2,
        ),
    ]

    reranked = reranker.rerank(query=query, matches=matches)
    assert len(reranked) == 2
    # c2 has exact term matches for water damage deductible, so reranker elevates it to rank 1
    assert reranked[0].chunk_id == "c2"
    assert reranked[0].rank == 1
    assert reranked[0].rerank_score > reranked[1].rerank_score
