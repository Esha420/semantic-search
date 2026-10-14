"""PostgreSQL Full-Text Search with fallback for keyword matching."""

from typing import Any
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.chunk import DocumentChunk
from src.models.document import Document
from src.schemas.retrieval import RetrievalFilter


async def search_full_text(
    session: AsyncSession,
    query_text: str,
    top_k: int = 10,
    filters: RetrievalFilter | None = None,
) -> list[dict[str, Any]]:
    """
    Executes keyword/full-text search using PostgreSQL tsvector and ts_rank_cd.
    Gracefully falls back to ILIKE if tsvector is unsupported (e.g. SQLite test env).
    """
    filters = filters or RetrievalFilter()

    try:
        # PostgreSQL tsvector full-text query
        ts_vector = func.to_tsvector("english", DocumentChunk.content)
        ts_query = func.plainto_tsquery("english", query_text)
        rank_expr = func.ts_rank_cd(ts_vector, ts_query)

        stmt = (
            select(
                DocumentChunk,
                Document.title.label("document_title"),
                Document.filename.label("filename"),
                rank_expr.label("rank_score"),
            )
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(
                DocumentChunk.tenant_id == filters.tenant_id,
                Document.status == "completed",
                ts_vector.op("@@")(ts_query),
            )
        )

        if filters.document_ids:
            stmt = stmt.where(DocumentChunk.document_id.in_(filters.document_ids))

        stmt = stmt.order_by(rank_expr.desc()).limit(top_k)
        result = await session.execute(stmt)
        rows = result.all()

        matches: list[dict[str, Any]] = []
        max_score = max([float(r[3]) for r in rows], default=1.0) or 1.0

        for rank, (chunk, doc_title, filename, score) in enumerate(rows, start=1):
            normalized_score = float(score) / max_score if max_score > 0 else 0.0
            matches.append({
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "document_title": doc_title,
                "filename": filename,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "dense_score": 0.0,
                "sparse_score": round(normalized_score, 4),
                "sparse_rank": rank,
                "metadata": chunk.metadata_json,
            })
        return matches

    except Exception:
        # Fallback for testing environments (e.g. SQLite or when FTS query fails)
        keywords = [k.lower() for k in query_text.split() if len(k) > 2]
        if not keywords:
            keywords = [query_text.lower()]

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
            )
        )
        if filters.document_ids:
            stmt = stmt.where(DocumentChunk.document_id.in_(filters.document_ids))

        result = await session.execute(stmt.limit(top_k * 3))
        all_rows = result.all()

        scored_rows: list[tuple[Any, str, str, float]] = []
        for chunk, doc_title, filename in all_rows:
            content_lower = chunk.content.lower()
            overlap = sum(1 for kw in keywords if kw in content_lower)
            if overlap > 0:
                score = overlap / len(keywords)
                scored_rows.append((chunk, doc_title, filename, score))

        scored_rows.sort(key=lambda x: x[3], reverse=True)
        top_rows = scored_rows[:top_k]

        matches = []
        for rank, (chunk, doc_title, filename, score) in enumerate(top_rows, start=1):
            matches.append({
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "document_title": doc_title,
                "filename": filename,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "dense_score": 0.0,
                "sparse_score": round(score, 4),
                "sparse_rank": rank,
                "metadata": chunk.metadata_json,
            })
        return matches
