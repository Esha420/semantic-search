"""SHA-256 Document hashing and idempotency detection."""

import hashlib
from typing import Literal
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.document import Document


def compute_sha256(content: bytes) -> str:
    """Computes SHA-256 hexadecimal digest for raw file bytes."""
    hasher = hashlib.sha256()
    hasher.update(content)
    return hasher.hexdigest()


async def inspect_document_idempotency(
    session: AsyncSession,
    tenant_id: str,
    filename: str,
    content_hash: str,
) -> tuple[Document | None, int, Literal["EXACT_MATCH", "NEW_VERSION", "NEW_DOCUMENT"]]:
    """
    Checks whether a document is an idempotent duplicate, an updated version, or completely new.

    Returns:
        (existing_doc, next_version_number, match_type)
    """
    # 1. Check for exact content hash match in this tenant
    hash_query = select(Document).where(
        Document.tenant_id == tenant_id,
        Document.content_hash == content_hash,
    )
    result = await session.execute(hash_query)
    exact_match = result.scalars().first()
    if exact_match and exact_match.status == "completed":
        return exact_match, exact_match.version, "EXACT_MATCH"

    # 2. Check if the same filename exists with an older hash (versioning)
    name_query = (
        select(Document)
        .where(Document.tenant_id == tenant_id, Document.filename == filename)
        .order_by(Document.version.desc())
    )
    name_result = await session.execute(name_query)
    latest_named_doc = name_result.scalars().first()

    if latest_named_doc:
        return latest_named_doc, latest_named_doc.version + 1, "NEW_VERSION"

    return None, 1, "NEW_DOCUMENT"
