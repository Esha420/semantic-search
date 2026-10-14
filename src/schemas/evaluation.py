"""Pydantic schemas for RAG Evaluation and Benchmarking."""

from typing import Any
from pydantic import BaseModel, Field


class BenchmarkTestCase(BaseModel):
    """Single test query in an evaluation benchmark suite."""
    query: str
    ground_truth_answer: str | None = None
    expected_doc_ids: list[str] = Field(default_factory=list)
    is_abstention_case: bool = False  # If True, system SHOULD abstain


class TestCaseResult(BaseModel):
    """Outcome of evaluating a single test case."""
    query: str
    retrieved_chunk_ids: list[str]
    retrieved_doc_ids: list[str]
    generated_answer: str
    is_abstained: bool
    expected_abstention: bool
    abstention_accurate: bool
    precision_at_k: float
    recall_at_k: float
    citations_count: int
    latency_ms: float


class BenchmarkEvalResult(BaseModel):
    """Aggregate benchmark results for the evaluation dataset."""
    total_queries: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    abstention_accuracy: float
    grounded_citation_rate: float
    average_latency_ms: float
    test_cases: list[TestCaseResult]
