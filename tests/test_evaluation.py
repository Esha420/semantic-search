"""Tests for RAG Evaluation and Benchmark calculation."""

import pytest
from src.evaluation.metrics import (
    aggregate_benchmark_results,
    calculate_precision_at_k,
    calculate_recall_at_k,
    evaluate_test_case,
)
from src.schemas.evaluation import BenchmarkTestCase
from src.schemas.rag import LatencyBreakdown, RAGResponse, TokenUsage
from src.schemas.retrieval import ChunkMatch


def test_precision_and_recall_metrics():
    retrieved = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
    expected = ["doc_2", "doc_3", "doc_9"]

    prec = calculate_precision_at_k(retrieved, expected)
    rec = calculate_recall_at_k(retrieved, expected)

    # 2 out of 5 retrieved are in expected -> 0.40
    assert prec == 0.4
    # 2 out of 3 expected were retrieved -> 0.6667
    assert rec == round(2 / 3, 4)


def test_evaluate_abstention_case():
    test_case = BenchmarkTestCase(
        query="What is the policy limit for Martian lunar colonies?",
        is_abstention_case=True,
    )

    rag_response = RAGResponse(
        query=test_case.query,
        answer="INSUFFICIENT_EVIDENCE: The documents do not contain evidence.",
        is_abstained=True,
        citations=[],
        model_name="mock-llm",
        latency=LatencyBreakdown(total_ms=45.0),
        usage=TokenUsage(),
        request_id="req_test",
        timestamp="2026-10-14T00:00:00Z",
    )

    res = evaluate_test_case(
        test_case=test_case,
        rag_response=rag_response,
        retrieved_chunks=[],
    )

    assert res.is_abstained is True
    assert res.expected_abstention is True
    assert res.abstention_accurate is True


def test_aggregate_benchmark_results():
    test_case1 = BenchmarkTestCase(query="Query 1", expected_doc_ids=["d1"], is_abstention_case=False)
    test_case2 = BenchmarkTestCase(query="Query 2", is_abstention_case=True)

    rag_res1 = RAGResponse(
        query="Query 1",
        answer="Answer with citation [1]",
        is_abstained=False,
        citations=[{"citation_id": 1, "document_id": "d1", "document_title": "D1", "filename": "f.txt", "page_number": 1, "chunk_index": 0, "snippet": "text", "relevance_score": 0.9}],
        model_name="mock-llm",
        latency=LatencyBreakdown(total_ms=50.0),
        usage=TokenUsage(),
        request_id="r1",
        timestamp="2026-10-14T00:00:00Z",
    )
    chunks1 = [
        ChunkMatch(chunk_id="c1", document_id="d1", document_title="D1", filename="f.txt", page_number=1, chunk_index=0, content="text", score=0.9, rank=1)
    ]
    r1 = evaluate_test_case(test_case1, rag_res1, chunks1)

    rag_res2 = RAGResponse(
        query="Query 2",
        answer="INSUFFICIENT_EVIDENCE",
        is_abstained=True,
        citations=[],
        model_name="mock-llm",
        latency=LatencyBreakdown(total_ms=30.0),
        usage=TokenUsage(),
        request_id="r2",
        timestamp="2026-10-14T00:00:00Z",
    )
    r2 = evaluate_test_case(test_case2, rag_res2, [])

    aggregated = aggregate_benchmark_results([r1, r2])
    assert aggregated.total_queries == 2
    assert aggregated.abstention_accuracy == 1.0
    assert aggregated.grounded_citation_rate == 1.0
    assert aggregated.average_latency_ms == 40.0
