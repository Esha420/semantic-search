"""End-to-end document ingestion pipeline."""

from typing import Any
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.embeddings.base import BaseEmbeddingProvider
from src.embeddings import get_embedding_provider
from src.ingestion.chunker import DocumentChunker
from src.ingestion.hasher import compute_sha256, inspect_document_idempotency
from src.ingestion.parser import DocumentParser
from src.logger import logger
from src.models.chunk import DocumentChunk
from src.models.document import Document


class IngestionPipeline:
    """Orchestrates parsing, hashing, chunking, embedding, and storage."""

    def __init__(
        self,
        chunker: DocumentChunker | None = None,
        embedding_provider: BaseEmbeddingProvider | None = None,
    ):
        self.chunker = chunker or DocumentChunker()
        self.embedding_provider = embedding_provider or get_embedding_provider()

    async def ingest_file(
        self,
        session: AsyncSession,
        filename: str,
        file_bytes: bytes,
        tenant_id: str = settings.DEFAULT_TENANT_ID,
        title: str | None = None,
        custom_metadata: dict[str, Any] | None = None,
    ) -> Document:
        """
        Executes idempotent document ingestion flow.

        1. Validates file size & format
        2. Computes content hash and detects duplicates
        3. Creates Document tracking record (status="processing")
        4. Extracts text pages (PDF/TXT/MD)
        5. Performs page-aware semantic chunking
        6. Computes vector embeddings in batches
        7. Persists chunks with vector and FTS tokens
        8. Finalizes Document status="completed"
        """
        # Validate file size
        file_size = len(file_bytes)
        if file_size > settings.MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"File size {file_size} bytes exceeds maximum allowed {settings.MAX_FILE_SIZE_BYTES} bytes."
            )

        doc_title = title or filename.rsplit(".", 1)[0].replace("_", " ").title()
        file_ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ".txt"

        # Content hash check for idempotency
        content_hash = compute_sha256(file_bytes)
        existing_doc, version_num, match_type = await inspect_document_idempotency(
            session=session,
            tenant_id=tenant_id,
            filename=filename,
            content_hash=content_hash,
        )

        if match_type == "EXACT_MATCH" and existing_doc:
            logger.info(
                f"Idempotent skip: Document {filename} with hash {content_hash[:8]} already completed."
            )
            return existing_doc

        # Create tracking record
        doc = Document(
            tenant_id=tenant_id,
            title=doc_title,
            filename=filename,
            file_type=file_ext,
            file_size_bytes=file_size,
            content_hash=content_hash,
            version=version_num,
            status="processing",
            metadata_json=custom_metadata or {},
        )
        session.add(doc)
        await session.commit()
        await session.refresh(doc)

        try:
            # Step 1: Parse pages
            pages = DocumentParser.parse(filename=filename, file_bytes=file_bytes)
            if not pages:
                raise ValueError("No extractable text content found in document.")

            # Step 2: Chunk pages preserving metadata
            chunks = self.chunker.chunk_pages(
                pages=pages,
                base_metadata={"document_id": doc.id, "filename": filename, "title": doc_title},
            )

            if not chunks:
                raise ValueError("Chunking produced 0 chunks from document.")

            # Step 3: Generate embeddings
            texts_to_embed = [c.content for c in chunks]
            embeddings = await self.embedding_provider.embed_texts(
                texts=texts_to_embed, input_type="passage"
            )

            # Step 4: Persist DocumentChunk records
            db_chunks: list[DocumentChunk] = []
            for chunk_data, emb in zip(chunks, embeddings, strict=True):
                db_chunk = DocumentChunk(
                    document_id=doc.id,
                    tenant_id=tenant_id,
                    chunk_index=chunk_data.chunk_index,
                    page_number=chunk_data.page_number,
                    content=chunk_data.content,
                    token_count=chunk_data.token_count,
                    embedding=emb,
                    metadata_json=chunk_data.metadata,
                )
                db_chunks.append(db_chunk)

            session.add_all(db_chunks)

            # Step 5: Mark document completed
            doc.status = "completed"
            doc.total_chunks = len(db_chunks)
            doc.error_message = None
            await session.commit()
            await session.refresh(doc)

            logger.info(
                f"Successfully ingested document '{doc.title}' (id={doc.id}, chunks={len(db_chunks)})"
            )
            return doc

        except Exception as exc:
            logger.error(f"Failed to ingest document {filename}: {exc}", exc_info=True)
            doc.status = "failed"
            doc.error_message = str(exc)
            await session.commit()
            raise
