"""Tests for Grounded RAG Generation, Citations, and Abstention."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.generation.engine import RAGEngine
from src.ingestion.pipeline import IngestionPipeline
from src.schemas.rag import RAGQueryRequest


@pytest.mark.asyncio
async def test_rag_grounded_answer_with_citations(
    db_session: AsyncSession,
    ingestion_pipeline: IngestionPipeline,
    rag_engine: RAGEngine,
):
    # Ingest document
    policy_text = (
        "Under Section 2.3, the commercial water damage deductible is set at $15,000 per occurrence. "
        "Claims must be reported within 30 calendar days of occurrence."
    )
    await ingestion_pipeline.ingest_file(
        session=db_session,
        filename="policy.txt",
        file_bytes=policy_text.encode("utf-8"),
        tenant_id="tenant_test",
        title="Commercial Property Policy",
    )

    # Query RAG Engine
    req = RAGQueryRequest(query="What is the commercial water damage deductible?")
    req.filters.tenant_id = "tenant_test"

    response = await rag_engine.answer_query(session=db_session, request=req)

    assert response.is_abstained is False
    assert response.abstention_reason is None
    assert len(response.citations) > 0

    # Verify citation fields
    citation = response.citations[0]
    assert citation.citation_id == 1
    assert citation.filename == "policy.txt"
    assert citation.page_number == 1
    assert "water damage deductible" in citation.snippet.lower()


@pytest.mark.asyncio
async def test_rag_abstains_when_no_documents_found(
    db_session: AsyncSession,
    rag_engine: RAGEngine,
):
    # Empty corpus query
    req = RAGQueryRequest(query="What is the deductible for Martian space equipment?")
    req.filters.tenant_id = "tenant_empty"

    response = await rag_engine.answer_query(session=db_session, request=req)

    assert response.is_abstained is True
    assert response.abstention_reason is not None
    assert response.abstention_reason.code == "NO_DOCUMENTS_FOUND"
    assert len(response.citations) == 0


@pytest.mark.asyncio
async def test_rag_abstains_on_low_confidence_query(
    db_session: AsyncSession,
    ingestion_pipeline: IngestionPipeline,
    rag_engine: RAGEngine,
):
    # Ingest insurance policy
    policy_text = "Fire coverage is $5,000,000. Sprinkler leaks are covered."
    await ingestion_pipeline.ingest_file(
        session=db_session,
        filename="policy.txt",
        file_bytes=policy_text.encode("utf-8"),
        tenant_id="tenant_test",
    )

    # Completely unrelated query with low similarity score triggers early abstention
    req = RAGQueryRequest(query="unrelated quantum computing algorithms in year 2050")
    req.filters.tenant_id = "tenant_test"

    response = await rag_engine.answer_query(session=db_session, request=req)

    assert response.is_abstained is True
    assert response.abstention_reason is not None
    assert response.abstention_reason.code in ("LOW_RETRIEVAL_CONFIDENCE", "INSUFFICIENT_EVIDENCE")


@pytest.mark.asyncio
async def test_rag_abstains_on_llm_insufficient_evidence(
    db_session: AsyncSession,
    ingestion_pipeline: IngestionPipeline,
    rag_engine: RAGEngine,
):
    # Ingest text that matches the query terms enough to pass retrieval, but LLM mock detects unsupported answer
    policy_text = (
        "Fire and hazard damage coverage terms and liability limits for commercial buildings."
    )
    await ingestion_pipeline.ingest_file(
        session=db_session,
        filename="policy.txt",
        file_bytes=policy_text.encode("utf-8"),
        tenant_id="tenant_test",
    )

    # Query with matching keywords to pass retrieval threshold, but asking unsupported question
    req = RAGQueryRequest(
        query="Fire and hazard damage unsupported_question about nuclear fusion power"
    )
    req.filters.tenant_id = "tenant_test"

    response = await rag_engine.answer_query(session=db_session, request=req)

    assert response.is_abstained is True
    assert response.abstention_reason is not None
    assert response.abstention_reason.code in ("INSUFFICIENT_EVIDENCE", "LOW_RETRIEVAL_CONFIDENCE")
