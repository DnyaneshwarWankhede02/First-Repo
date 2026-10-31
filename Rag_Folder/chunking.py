"""
Chunker: token-based sliding-window chunking with overlap.

Chunks are assembled from sentence units so they end at natural boundaries,
and each chunk records the page number of its first sentence, preserving page
provenance even when a chunk spans a page break.

Token counting uses tiktoken (cl100k_base) when installed; otherwise it falls
back to a words-based estimate (~0.75 words per token), which is accurate
enough for the "approximately 500 tokens" target.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple

from config import get_logger

logger = get_logger("chunker")

try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")
    _TOKENIZER = "tiktoken cl100k_base"

    def count_tokens(text: str, words_per_token: float = 0.75) -> int:
        return len(_ENCODING.encode(text))
except Exception:
    _ENCODING = None
    _TOKENIZER = "words/0.75 estimate"

    def count_tokens(text: str, words_per_token: float = 0.75) -> int:
        return max(1, int(round(len(text.split()) / words_per_token)))


# Split on sentence enders followed by the start of a new sentence
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


@dataclass(frozen=True)
class Chunk:
    """One text chunk with its position and size metadata."""
    chunk_index: int
    chunk_text: str
    page_number: int
    token_count: int


def _split_long_sentence(sentence: str, max_tokens: int) -> List[str]:
    """Break a sentence longer than a whole chunk into word-bounded pieces."""
    words = sentence.split()
    pieces, current, current_cost = [], [], 0
    for word in words:
        cost = max(1, len(word) // 4)  # cheap per-word token approximation
        if current and current_cost + cost > max_tokens:
            pieces.append(" ".join(current))
            current, current_cost = [word], cost
        else:
            current.append(word)
            current_cost += cost
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_pages(pages, chunk_size_tokens: int = 500,
               chunk_overlap_tokens: int = 100,
               words_per_token: float = 0.75) -> List[Chunk]:
    """Convert parsed PageText pages into overlapping, token-bounded chunks."""
    if not pages:
        return []

    # 1) Build (sentence, page_number) units from the cleaned page texts.
    units: List[Tuple[str, int]] = []
    for page in pages:
        for sentence in _SENTENCE_SPLIT.split(page.text):
            sentence = sentence.strip()
            if not sentence:
                continue
            if count_tokens(sentence, words_per_token) > chunk_size_tokens:
                for piece in _split_long_sentence(sentence, chunk_size_tokens):
                    units.append((piece, page.page_number))
            else:
                units.append((sentence, page.page_number))

    # 2) Greedy sliding window: fill a chunk up to the token budget, then
    #    restart ~chunk_overlap_tokens earlier.
    chunks: List[Chunk] = []
    i = 0
    while i < len(units):
        chunk_units, total = [], 0
        j = i
        while j < len(units):
            t = count_tokens(units[j][0], words_per_token)
            if chunk_units and total + t > chunk_size_tokens:
                break
            chunk_units.append(units[j])
            total += t
            j += 1

        chunk_text = " ".join(u[0] for u in chunk_units)

        # Skip a trailing chunk that is fully contained in the previous chunk
        # (can happen when the last units are shorter than the overlap).
        if chunks and chunks[-1].chunk_text.endswith(chunk_text):
            pass
        else:
            chunks.append(Chunk(
                chunk_index=len(chunks),
                chunk_text=chunk_text,
                page_number=chunk_units[0][1],
                token_count=total,
            ))

        # Back the start point up so the next chunk overlaps ~chunk_overlap_tokens
        next_i = j
        tail = 0
        while next_i > i + 1 and tail < chunk_overlap_tokens:
            next_i -= 1
            tail += count_tokens(chunk_units[next_i - i][0], words_per_token)
        i = next_i

    logger.info("Produced %d chunks (target %d tokens, overlap %d) using %s",
                len(chunks), chunk_size_tokens, chunk_overlap_tokens, _TOKENIZER)
    return chunks
