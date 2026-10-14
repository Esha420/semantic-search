"""Tests for SHA-256 document hashing, idempotency, and version tracking."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.ingestion.hasher import compute_sha256, inspect_document_idempotency
from src.models.document import Document


def test_compute_sha256():
    data = b"Verisk Property Casualty 2026"
    hash1 = compute_sha256(data)
    hash2 = compute_sha256(data)
    assert hash1 == hash2
    assert len(hash1) == 64
    assert hash1 != compute_sha256(b"Different content")


@pytest.mark.asyncio
async def test_idempotency_exact_match(db_session: AsyncSession):
    content = b"Policy Document v1"
    content_hash = compute_sha256(content)

    # Insert existing completed document
    doc = Document(
        tenant_id="tenant_test",
        title="Policy Document",
        filename="policy.txt",
        file_type=".txt",
        file_size_bytes=len(content),
        content_hash=content_hash,
        version=1,
        status="completed",
    )
    db_session.add(doc)
    await db_session.commit()

    # Inspect identical document upload
    matched_doc, version_num, match_type = await inspect_document_idempotency(
        session=db_session,
        tenant_id="tenant_test",
        filename="policy.txt",
        content_hash=content_hash,
    )
    assert match_type == "EXACT_MATCH"
    assert matched_doc.id == doc.id
    assert version_num == 1


@pytest.mark.asyncio
async def test_idempotency_new_version(db_session: AsyncSession):
    content_v1 = b"Policy Document v1"
    hash_v1 = compute_sha256(content_v1)

    doc_v1 = Document(
        tenant_id="tenant_test",
        title="Policy Document",
        filename="policy.txt",
        file_type=".txt",
        file_size_bytes=len(content_v1),
        content_hash=hash_v1,
        version=1,
        status="completed",
    )
    db_session.add(doc_v1)
    await db_session.commit()

    # Upload new content under the same filename
    content_v2 = b"Policy Document v2 Updated Deductibles"
    hash_v2 = compute_sha256(content_v2)

    matched_doc, version_num, match_type = await inspect_document_idempotency(
        session=db_session,
        tenant_id="tenant_test",
        filename="policy.txt",
        content_hash=hash_v2,
    )
    assert match_type == "NEW_VERSION"
    assert version_num == 2
