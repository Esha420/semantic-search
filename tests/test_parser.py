"""Tests for Document Parser with PyMuPDF and text extraction."""

import pymupdf
import pytest
from src.ingestion.parser import DocumentParser, ExtractedPage


def test_parse_plain_text():
    sample_text = b"This is a test plain text document for Verisk Document Intelligence."
    pages = DocumentParser.parse_plain_text(sample_text)
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "Verisk Document Intelligence" in pages[0].text


def test_parse_pdf_in_memory():
    # Create an in-memory 2-page PDF using pymupdf
    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_text((50, 72), "Section 1: Commercial Property Limits and Perils")
    page2 = doc.new_page()
    page2.insert_text((50, 72), "Section 2: Deductibles and Exclusions for Flood")
    pdf_bytes = doc.tobytes()
    doc.close()

    pages = DocumentParser.parse(filename="sample_policy.pdf", file_bytes=pdf_bytes)
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert "Commercial Property Limits" in pages[0].text
    assert pages[1].page_number == 2
    assert "Deductibles and Exclusions" in pages[1].text


def test_parse_dispatch_by_extension():
    content = b"# Policy Header\nBody of markdown file."
    pages = DocumentParser.parse(filename="terms.md", file_bytes=content)
    assert len(pages) == 1
    assert "Policy Header" in pages[0].text
