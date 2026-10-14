"""Document parser supporting PDF, TXT, and Markdown extraction with page numbers."""

import io
from dataclasses import dataclass
from typing import BinaryIO
import pymupdf

from src.logger import logger


@dataclass
class ExtractedPage:
    """Extracted text content associated with a page number."""
    page_number: int
    text: str


class DocumentParser:
    """Robust document extractor preserving page structure for verifiable citations."""

    @staticmethod
    def parse_pdf(file_bytes: bytes) -> list[ExtractedPage]:
        """Extracts text per page from a PDF byte stream using PyMuPDF."""
        pages: list[ExtractedPage] = []
        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            for page_index in range(len(doc)):
                page = doc[page_index]
                text = page.get_text("text").strip()
                if text:
                    pages.append(
                        ExtractedPage(page_number=page_index + 1, text=text)
                    )
            doc.close()
        except Exception as exc:
            logger.error(f"Error parsing PDF with PyMuPDF: {exc}")
            raise ValueError(f"Failed to extract text from PDF: {exc}") from exc

        return pages

    @staticmethod
    def parse_plain_text(file_bytes: bytes) -> list[ExtractedPage]:
        """Extracts text from plain text or Markdown files."""
        try:
            # Try UTF-8 first, fallback to latin-1
            try:
                content = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                content = file_bytes.decode("latin-1")

            cleaned = content.strip()
            if not cleaned:
                return []

            # For single text files, we treat the entire file as page 1
            return [ExtractedPage(page_number=1, text=cleaned)]
        except Exception as exc:
            logger.error(f"Error parsing plain text: {exc}")
            raise ValueError(f"Failed to extract text: {exc}") from exc

    @classmethod
    def parse(cls, filename: str, file_bytes: bytes) -> list[ExtractedPage]:
        """Dispatches parser based on file extension."""
        lower_name = filename.lower()
        if lower_name.endswith(".pdf"):
            return cls.parse_pdf(file_bytes)
        elif lower_name.endswith((".txt", ".md", ".json", ".csv")):
            return cls.parse_plain_text(file_bytes)
        else:
            # Fallback attempt as text
            return cls.parse_plain_text(file_bytes)
