"""Automated RAG evaluation and benchmark endpoint."""

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_tenant, get_db, verify_api_key
from src.evaluation.metrics import aggregate_benchmark_results, evaluate_test_case
from src.generation.engine import RAGEngine
from src.retrieval.engine import RetrievalEngine
from src.schemas.evaluation import BenchmarkEvalResult, BenchmarkTestCase, TestCaseResult
from src.schemas.rag import RAGQueryRequest
from src.schemas.retrieval import RetrievalQuery

router = APIRouter(prefix="/evaluation", tags=["Evaluation & Benchmarks"])


@router.post(
    "/run",
    response_model=BenchmarkEvalResult,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)],
)
async def run_benchmark(
    test_cases: list[BenchmarkTestCase],
    tenant_id: str = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
) -> BenchmarkEvalResult:
    """
    Executes automated RAG evaluation over a suite of test cases:
    - Measures Precision@K and Recall@K against expected documents
    - Tests answer grounding and citation preservation
    - Validates abstention behavior on questions without supporting evidence
    """
    rag_engine = RAGEngine()
    retrieval_engine = RetrievalEngine()
    results: list[TestCaseResult] = []

    for case in test_cases:
        # 1. Retrieve candidates
        ret_query = RetrievalQuery(query=case.query, top_k=5)
        ret_query.filters.tenant_id = tenant_id
        ret_res = await retrieval_engine.search(session=db, query=ret_query)

        # 2. Query RAG engine
        rag_req = RAGQueryRequest(query=case.query, top_k=5)
        rag_req.filters.tenant_id = tenant_id
        rag_res = await rag_engine.answer_query(session=db, request=rag_req)

        # 3. Calculate metrics
        eval_res = evaluate_test_case(
            test_case=case,
            rag_response=rag_res,
            retrieved_chunks=ret_res.matches,
        )
        results.append(eval_res)

    return aggregate_benchmark_results(results)
