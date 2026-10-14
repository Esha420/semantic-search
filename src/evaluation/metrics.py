"""Evaluation metrics for measuring RAG quality, precision, recall, and abstention."""

from typing import Any
from src.schemas.evaluation import BenchmarkEvalResult, BenchmarkTestCase, TestCaseResult
from src.schemas.rag import RAGResponse
from src.schemas.retrieval import ChunkMatch


def calculate_precision_at_k(retrieved_doc_ids: list[str], expected_doc_ids: list[str]) -> float:
    """Calculates Precision@K = |Retrieved relevant docs| / |Total retrieved docs|."""
    if not retrieved_doc_ids:
        return 0.0
    if not expected_doc_ids:
        return 0.0
    relevant_hits = sum(1 for d in retrieved_doc_ids if d in expected_doc_ids)
    return round(relevant_hits / len(retrieved_doc_ids), 4)


def calculate_recall_at_k(retrieved_doc_ids: list[str], expected_doc_ids: list[str]) -> float:
    """Calculates Recall@K = |Unique retrieved relevant docs| / |Expected docs|."""
    if not expected_doc_ids:
        return 1.0 if not retrieved_doc_ids else 0.0
    unique_hits = len(set(retrieved_doc_ids).intersection(set(expected_doc_ids)))
    return round(unique_hits / len(set(expected_doc_ids)), 4)


def evaluate_test_case(
    test_case: BenchmarkTestCase,
    rag_response: RAGResponse,
    retrieved_chunks: list[ChunkMatch],
) -> TestCaseResult:
    """Evaluates a single query against ground truth expectations."""
    retrieved_chunk_ids = [c.chunk_id for c in retrieved_chunks]
    retrieved_doc_ids = [c.document_id for c in retrieved_chunks]

    prec = calculate_precision_at_k(retrieved_doc_ids, test_case.expected_doc_ids)
    rec = calculate_recall_at_k(retrieved_doc_ids, test_case.expected_doc_ids)

    # Check abstention correctness
    is_abstained = rag_response.is_abstained
    abstention_accurate = (is_abstained == test_case.is_abstention_case)

    return TestCaseResult(
        query=test_case.query,
        retrieved_chunk_ids=retrieved_chunk_ids,
        retrieved_doc_ids=retrieved_doc_ids,
        generated_answer=rag_response.answer,
        is_abstained=is_abstained,
        expected_abstention=test_case.is_abstention_case,
        abstention_accurate=abstention_accurate,
        precision_at_k=prec,
        recall_at_k=rec,
        citations_count=len(rag_response.citations),
        latency_ms=rag_response.latency.total_ms,
    )


def aggregate_benchmark_results(results: list[TestCaseResult]) -> BenchmarkEvalResult:
    """Aggregates multiple test case outcomes into overall benchmark KPIs."""
    if not results:
        return BenchmarkEvalResult(
            total_queries=0,
            mean_precision_at_k=0.0,
            mean_recall_at_k=0.0,
            abstention_accuracy=0.0,
            grounded_citation_rate=0.0,
            average_latency_ms=0.0,
            test_cases=[],
        )

    n = len(results)
    mean_prec = round(sum(r.precision_at_k for r in results) / n, 4)
    mean_rec = round(sum(r.recall_at_k for r in results) / n, 4)
    abstention_acc = round(sum(1 for r in results if r.abstention_accurate) / n, 4)

    # Citation rate for non-abstained queries
    non_abstained = [r for r in results if not r.expected_abstention]
    if non_abstained:
        cited_count = sum(1 for r in non_abstained if r.citations_count > 0)
        citation_rate = round(cited_count / len(non_abstained), 4)
    else:
        citation_rate = 1.0

    avg_lat = round(sum(r.latency_ms for r in results) / n, 2)

    return BenchmarkEvalResult(
        total_queries=n,
        mean_precision_at_k=mean_prec,
        mean_recall_at_k=mean_rec,
        abstention_accuracy=abstention_acc,
        grounded_citation_rate=citation_rate,
        average_latency_ms=avg_lat,
        test_cases=results,
    )
