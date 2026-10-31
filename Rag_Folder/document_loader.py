"""
Document loader: discovers and validates source PDFs inside a Unity Catalog Volume.

The loader is deliberately generic - point IngestionConfig.volume_path at any
volume and it will pick up every file matching the configured extensions.
Zero-byte or unreadable files are logged and skipped instead of failing the run.
"""

import os
from typing import Dict, List

from config import IngestionConfig, get_logger

logger = get_logger("loader")


def list_documents(cfg: IngestionConfig) -> List[Dict]:
    """Return metadata for every readable source document in the volume.

    Each entry: {file_name, file_path, file_size_bytes, document_type}.
    Raises only when the volume itself is missing/unlistable.
    """
    volume = cfg.volume_path
    if not os.path.isdir(volume):
        logger.error("Volume path not found or not accessible: %s", volume)
        raise FileNotFoundError(f"Volume path not found: {volume}")

    try:
        entries = sorted(os.listdir(volume))
    except OSError as exc:
        logger.error("Cannot list volume %s: %s", volume, exc)
        raise

    documents: List[Dict] = []
    for name in entries:
        ext = os.path.splitext(name)[1].lower()
        if ext not in tuple(e.lower() for e in cfg.file_extensions):
            continue
        path = os.path.join(volume, name)
        if not os.path.isfile(path):
            continue
        try:
            size = os.path.getsize(path)
        except OSError as exc:
            logger.warning("Skipping %s: cannot stat file (%s)", name, exc)
            continue
        if size == 0:
            logger.warning("Skipping %s: file is empty (0 bytes)", name)
            continue
        documents.append({
            "file_name": name,
            "file_path": path,
            "file_size_bytes": size,
            "document_type": ext.lstrip("."),
        })
        logger.info("Found source document: %s (%.1f KB)", name, size / 1024.0)

    if documents:
        logger.info("Discovered %d document(s) in %s", len(documents), volume)
    else:
        logger.warning("No %s documents found in %s", tuple(cfg.file_extensions), volume)
    return documents
