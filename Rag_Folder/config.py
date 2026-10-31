"""
Central configuration for the Stage 1 document ingestion pipeline.

All tunables (source volume, destination table, chunking parameters) live here
so the pipeline can be re-targeted at any volume/table without touching the
module code.
"""

import logging
import sys
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class IngestionConfig:
    """Configuration for one ingestion run."""

    # --- Source: Unity Catalog Volume holding the raw documents ---
    volume_path: str = "/Volumes/dev_project/rag_docs/source_pdfs"

    # --- Destination: Unity Catalog Delta table holding the chunks ---
    catalog: str = "dev_project"
    schema_name: str = "rag_docs"
    table_name: str = "document_chunks"

    # --- Chunking ---
    chunk_size_tokens: int = 500       # approximate tokens per chunk
    chunk_overlap_tokens: int = 100     # approximate overlap between chunks

    # --- File selection ---
    file_extensions: Tuple[str, ...] = (".pdf",)

    # --- Token estimation fallback (used when tiktoken is unavailable) ---
    words_per_token: float = 0.75

    @property
    def full_table_name(self) -> str:
        return f"{self.catalog}.{self.schema_name}.{self.table_name}"


def get_logger(name: str) -> logging.Logger:
    """Return a console logger with a consistent format (no duplicate handlers)."""
    logger = logging.getLogger(f"rag_ingestion.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
