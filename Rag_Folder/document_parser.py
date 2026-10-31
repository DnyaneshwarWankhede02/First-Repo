"""
Document parser: extracts raw text from PDFs page by page with pypdf.

Every extracted page keeps its 1-based page_number so downstream chunks can
carry page provenance. Corrupted or password-protected files, empty pages, and
per-page extraction failures are logged and skipped gracefully - a bad file
never aborts the whole ingestion run.
"""

from dataclasses import dataclass
from typing import List

from pypdf import PdfReader

from config import get_logger

logger = get_logger("parser")


@dataclass(frozen=True)
class PageText:
    """Raw text extracted from a single page."""
    page_number: int
    text: str


def parse_pdf(file_path: str, document_name: str) -> List[PageText]:
    """Parse one PDF into a list of PageText records.

    Returns [] when the file cannot be opened (corrupt/encrypted); the caller
    treats that as a skipped document, not a pipeline failure.
    """
    try:
        reader = PdfReader(file_path, strict=False)
    except Exception as exc:  # corrupted or unreadable PDF
        logger.error("Skipping '%s': cannot open PDF (%s: %s)",
                     document_name, type(exc).__name__, exc)
        return []

    if getattr(reader, "is_encrypted", False):
        try:
            if not reader.decrypt(""):
                logger.error("Skipping '%s': PDF is password-protected", document_name)
                return []
        except Exception as exc:
            logger.error("Skipping '%s': encrypted PDF cannot be decrypted (%s)",
                         document_name, exc)
            return []

    pages: List[PageText] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:
            logger.warning("'%s' page %d: extraction failed (%s: %s) - page skipped",
                           document_name, page_number, type(exc).__name__, exc)
            continue
        if not raw_text.strip():
            logger.info("'%s' page %d: empty page - skipped", document_name, page_number)
            continue
        pages.append(PageText(page_number=page_number, text=raw_text))

    logger.info("Parsed '%s': %d/%d pages yielded text",
                document_name, len(pages), len(reader.pages))
    return pages
