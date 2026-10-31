"""
Ingestion pipeline (Stage 1): load -> parse -> clean -> chunk -> store.

Idempotency & incremental ingestion:
  * document_id = UUIDv5(document_name)              -> stable per document
  * chunk_id    = UUIDv5(document_id, chunk_index)   -> stable per chunk position
  * Chunks are MERGEd on chunk_id, so repeated runs update existing rows
    instead of duplicating them, and chunks that disappeared from a
    re-parsed document are deleted.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from pyspark.sql import SparkSession
from pyspark.sql.types import (IntegerType, StringType, StructField, StructType,
                               TimestampType)

from chunking import chunk_pages
from config import IngestionConfig, get_logger
from document_loader import list_documents
from document_parser import PageText, parse_pdf
from text_cleaner import clean_text

logger = get_logger("pipeline")

CHUNK_SCHEMA = StructType([
    StructField("document_id", StringType(), nullable=False),
    StructField("document_name", StringType(), nullable=True),
    StructField("document_type", StringType(), nullable=True),
    StructField("chunk_id", StringType(), nullable=False),
    StructField("chunk_index", IntegerType(), nullable=False),
    StructField("chunk_text", StringType(), nullable=True),
    StructField("page_number", IntegerType(), nullable=True),
    StructField("token_count", IntegerType(), nullable=True),
    StructField("created_at", TimestampType(), nullable=True),
])


def make_document_id(document_name: str) -> str:
    """Deterministic document ID derived from the file name (stable across runs)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, document_name))


def make_chunk_id(document_id: str, chunk_index: int) -> str:
    """Deterministic chunk ID derived from (document_id, chunk_index)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}::chunk_{chunk_index}"))


def ensure_table(cfg: IngestionConfig, spark: SparkSession) -> None:
    """Create the schema and destination Delta table when they do not exist."""
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.catalog}.{cfg.schema_name}")
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {cfg.full_table_name} (
            document_id   STRING  COMMENT 'Stable UUID per document name',
            document_name STRING,
            document_type STRING,
            chunk_id      STRING  COMMENT 'Stable UUID per (document, chunk index)',
            chunk_index   INT     COMMENT '0-based position of the chunk within its document',
            chunk_text    STRING,
            page_number   INT     COMMENT '1-based page number where the chunk starts',
            token_count   INT,
            created_at    TIMESTAMP
        )
    """)
    logger.info("Destination table ready: %s", cfg.full_table_name)


def build_chunk_records(cfg: IngestionConfig) -> Tuple[List[Dict], List[Dict]]:
    """Load, parse, clean, and chunk every source document into record dicts."""
    documents = list_documents(cfg)
    records: List[Dict] = []
    doc_summaries: List[Dict] = []

    for doc in documents:
        document_id = make_document_id(doc["file_name"])

        pages = parse_pdf(doc["file_path"], doc["file_name"])
        if not pages:
            logger.warning("No text extracted from '%s' - document skipped",
                           doc["file_name"])
            continue

        cleaned_pages = [PageText(p.page_number, clean_text(p.text)) for p in pages]
        cleaned_pages = [p for p in cleaned_pages if p.text]

        chunks = chunk_pages(cleaned_pages,
                             chunk_size_tokens=cfg.chunk_size_tokens,
                             chunk_overlap_tokens=cfg.chunk_overlap_tokens,
                             words_per_token=cfg.words_per_token)

        for chunk in chunks:
            records.append({
                "document_id": document_id,
                "document_name": doc["file_name"],
                "document_type": doc["document_type"],
                "chunk_id": make_chunk_id(document_id, chunk.chunk_index),
                "chunk_index": chunk.chunk_index,
                "chunk_text": chunk.chunk_text,
                "page_number": chunk.page_number,
                "token_count": chunk.token_count,
                "created_at": datetime.now(timezone.utc),
            })

        doc_summaries.append({
            "document_id": document_id,
            "document_name": doc["file_name"],
            "pages_parsed": len(pages),
            "chunks_created": len(chunks),
        })
        logger.info("Prepared %d chunks from '%s'", len(chunks), doc["file_name"])

    return records, doc_summaries


def write_chunks(cfg: IngestionConfig, records: List[Dict],
                 spark: SparkSession) -> Dict[str, int]:
    """MERGE chunk records into the Delta table (idempotent) and prune stale chunks."""
    if not records:
        logger.warning("No chunk records to write")
        return {"rows_written": 0}

    # createDataFrame with an explicit schema requires tuples, not dicts
    rows = [(r["document_id"], r["document_name"], r["document_type"], r["chunk_id"],
             r["chunk_index"], r["chunk_text"], r["page_number"], r["token_count"],
             r["created_at"]) for r in records]
    df = spark.createDataFrame(rows, schema=CHUNK_SCHEMA)
    df.createOrReplaceTempView("stage1_new_chunks")

    # Upsert: matched rows are refreshed (original created_at preserved),
    # new rows inserted.
    spark.sql(f"""
        MERGE INTO {cfg.full_table_name} AS t
        USING stage1_new_chunks AS s
        ON t.chunk_id = s.chunk_id
        WHEN MATCHED THEN UPDATE SET
            t.document_id   = s.document_id,
            t.document_name = s.document_name,
            t.document_type = s.document_type,
            t.chunk_index   = s.chunk_index,
            t.chunk_text    = s.chunk_text,
            t.page_number   = s.page_number,
            t.token_count   = s.token_count
        WHEN NOT MATCHED THEN INSERT *
    """)

    # Prune chunks that no longer exist after a document changed or shrank.
    spark.sql(f"""
        DELETE FROM {cfg.full_table_name}
        WHERE document_id IN (SELECT DISTINCT document_id FROM stage1_new_chunks)
          AND chunk_id NOT IN (SELECT chunk_id FROM stage1_new_chunks)
    """)

    return {"rows_written": len(records)}


def run_ingestion(cfg: IngestionConfig, spark: SparkSession) -> Dict[str, Any]:
    """Execute one full ingestion run and return a summary dict."""
    logger.info("=== Stage 1 ingestion started ===")
    logger.info("Source volume : %s", cfg.volume_path)
    logger.info("Target table  : %s", cfg.full_table_name)

    ensure_table(cfg, spark)
    records, doc_summaries = build_chunk_records(cfg)
    write_result = write_chunks(cfg, records, spark)

    summary = {
        "documents_processed": len(doc_summaries),
        "chunks_prepared": len(records),
        "documents": doc_summaries,
        **write_result,
    }
    logger.info("=== Stage 1 ingestion finished: %d document(s), %d chunk(s) ===",
                summary["documents_processed"], summary["chunks_prepared"])
    return summary
