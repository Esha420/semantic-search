"""Vector similarity search using pgvector cosine distance with fallback for SQLite tests."""

import math
from typing import Any
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.chunk import DocumentChunk
from src.models.document import Document
from src.schemas.retrieval import RetrievalFilter


def _python_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculates cosine similarity in python memory."""
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


async def search_vector_similarity(
    session: AsyncSession,
    query_vector: list[float],
    top_k: int = 10,
    filters: RetrievalFilter | None = None,
) -> list[dict[str, Any]]:
    """
    Executes dense approximate nearest neighbor search via pgvector.
    Falls back gracefully to in-memory cosine computation on non-Postgres (e.g. SQLite tests).
    """
    filters = filters or RetrievalFilter()

    # Check dialect
    bind = session.bind
    is_sqlite = bind and bind.dialect.name == "sqlite"

    if not is_sqlite:
        try:
            # PostgreSQL pgvector native search
            distance_expr = DocumentChunk.embedding.cosine_distance(query_vector)

            query = (
                select(
                    DocumentChunk,
                    Document.title.label("document_title"),
                    Document.filename.label("filename"),
                    distance_expr.label("distance"),
                )
                .join(Document, DocumentChunk.document_id == Document.id)
                .where(
                    DocumentChunk.tenant_id == filters.tenant_id,
                    Document.status == "completed",
                    DocumentChunk.embedding.isnot(None),
                )
            )

            if filters.document_ids:
                query = query.where(DocumentChunk.document_id.in_(filters.document_ids))

            query = query.order_by(distance_expr.asc()).limit(top_k)

            result = await session.execute(query)
            rows = result.all()

            matches: list[dict[str, Any]] = []
            for rank, (chunk, doc_title, filename, dist) in enumerate(rows, start=1):
                similarity = max(0.0, 1.0 - float(dist)) if dist is not None else 0.0
                matches.append({
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "document_title": doc_title,
                    "filename": filename,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "dense_score": round(similarity, 4),
                    "sparse_score": 0.0,
                    "dense_rank": rank,
                    "metadata": chunk.metadata_json,
                })

            return matches
        except Exception:
            # If native pgvector operator failed, fall through to in-memory fallback
            pass

    # SQLite fallback for lightning-fast test suites
    stmt = (
        select(
            DocumentChunk,
            Document.title.label("document_title"),
            Document.filename.label("filename"),
        )
        .join(Document, DocumentChunk.document_id == Document.id)
        .where(
            DocumentChunk.tenant_id == filters.tenant_id,
            Document.status == "completed",
            DocumentChunk.embedding.isnot(None),
        )
    )

    if filters.document_ids:
        stmt = stmt.where(DocumentChunk.document_id.in_(filters.document_ids))

    res = await session.execute(stmt)
    all_chunks = res.all()

    scored = []
    for chunk, doc_title, filename in all_chunks:
        emb = chunk.embedding
        if isinstance(emb, str):
            import json
            try:
                emb = json.loads(emb)
            except Exception:
                emb = []
        score = _python_cosine_similarity(emb, query_vector)
        scored.append((chunk, doc_title, filename, score))

    scored.sort(key=lambda x: x[3], reverse=True)
    top_matches = scored[:top_k]

    matches = []
    for rank, (chunk, doc_title, filename, score) in enumerate(top_matches, start=1):
        matches.append({
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "document_title": doc_title,
            "filename": filename,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "dense_score": round(score, 4),
            "sparse_score": 0.0,
            "dense_rank": rank,
            "metadata": chunk.metadata_json,
        })

    return matches
