"""Tests for Semantic Document Chunker preserving page numbers and citation tokens."""

from src.ingestion.chunker import DocumentChunker, estimate_token_count
from src.ingestion.parser import ExtractedPage


def test_estimate_token_count():
    text = "Underwriting evaluation for commercial properties in high flood zones."
    tokens = estimate_token_count(text)
    assert tokens > 5


def test_chunk_pages_preserves_page_numbers():
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
    pages = [
        ExtractedPage(page_number=1, text="Page 1: Overview of commercial property perils and limits."),
        ExtractedPage(page_number=2, text="Page 2: Standard deductible is $10,000 per occurrence for fire."),
        ExtractedPage(page_number=3, text="Page 3: Coinsurance requirements and roof age limits."),
    ]

    chunks = chunker.chunk_pages(pages=pages, base_metadata={"test": True})
    assert len(chunks) >= 3

    # Check page preservation
    page_numbers = [c.page_number for c in chunks]
    assert 1 in page_numbers
    assert 2 in page_numbers
    assert 3 in page_numbers

    # Verify monotonic indexing
    for idx, c in enumerate(chunks):
        assert c.chunk_index == idx
        assert c.token_count > 0
