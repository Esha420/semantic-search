"""Celery background tasks for asynchronous document ingestion."""

import asyncio
from src.database import async_session_maker
from src.ingestion.pipeline import IngestionPipeline
from src.logger import logger
from src.tasks.celery_app import celery_app


async def _async_ingest_wrapper(
    filename: str,
    file_bytes: bytes,
    tenant_id: str,
    title: str | None,
):
    async with async_session_maker() as session:
        pipeline = IngestionPipeline()
        await pipeline.ingest_file(
            session=session,
            filename=filename,
            file_bytes=file_bytes,
            tenant_id=tenant_id,
            title=title,
        )


@celery_app.task(name="tasks.ingest_document", bind=True, max_retries=3)
def ingest_document_task(
    self,
    filename: str,
    file_bytes_hex: str,
    tenant_id: str,
    title: str | None = None,
):
    """Celery background task for parsing, chunking, and embedding."""
    try:
        logger.info(f"Starting Celery background ingestion for '{filename}' (tenant={tenant_id})...")
        file_bytes = bytes.fromhex(file_bytes_hex)
        asyncio.run(
            _async_ingest_wrapper(
                filename=filename,
                file_bytes=file_bytes,
                tenant_id=tenant_id,
                title=title,
            )
        )
        logger.info(f"Celery ingestion completed for '{filename}'.")
        return {"status": "completed", "filename": filename}
    except Exception as exc:
        logger.error(f"Celery task failed for {filename}: {exc}", exc_info=True)
        raise self.retry(exc=exc, countdown=10 * (self.request.retries + 1))
