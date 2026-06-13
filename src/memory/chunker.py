"""
Chunker: text -> overlapping chunks with metadata.

Why chunk at all: embeddings have a fixed input window and a single vector
summarizes a whole chunk, so huge chunks blur meaning while tiny chunks lose
context. Overlap keeps a fact that straddles a boundary recoverable.

This is a recursive splitter: it prefers to break on paragraph, then sentence,
then word boundaries, and approximates token count as ~0.75 * word_count
(good enough for sizing; swap in tiktoken if you want exactness). Kept fully
deterministic so tests/test_chunker.py can assert on it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.config import settings

_WORDS_PER_TOKEN = 0.75  # ~1.33 tokens per word for English


def _approx_tokens(text: str) -> int:
    words = len(text.split())
    return max(1, round(words / _WORDS_PER_TOKEN))


@dataclass
class Chunk:
    content: str
    token_count: int
    metadata: dict = field(default_factory=dict)


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _split_words(text: str, max_tokens: int) -> list[str]:
    """Last-resort splitter for a unit with no paragraph/sentence boundaries."""
    words = text.split()
    max_words = max(1, int(max_tokens * _WORDS_PER_TOKEN))
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
    metadata: dict | None = None,
) -> list[Chunk]:
    chunk_size = chunk_size or settings.chunk_size_tokens
    overlap = overlap or settings.chunk_overlap_tokens
    metadata = metadata or {}

    text = re.sub(r"[ \t]+", " ", text or "").strip()
    if not text:
        return []

    # Build candidate units (paragraphs, then split oversized paragraphs into
    # sentences, then split still-oversized sentences by words) so no single
    # unit ever exceeds the chunk budget.
    units: list[str] = []
    for para in _split_paragraphs(text):
        if _approx_tokens(para) <= chunk_size:
            units.append(para)
            continue
        for sent in _split_sentences(para):
            if _approx_tokens(sent) <= chunk_size:
                units.append(sent)
            else:
                units.extend(_split_words(sent, chunk_size))

    chunks: list[Chunk] = []
    current: list[str] = []
    current_tokens = 0

    def flush():
        nonlocal current, current_tokens
        if not current:
            return
        joined = " ".join(current).strip()
        chunks.append(
            Chunk(content=joined, token_count=_approx_tokens(joined), metadata=dict(metadata))
        )

    for unit in units:
        ut = _approx_tokens(unit)
        if current_tokens + ut > chunk_size and current:
            flush()
            # Build overlap tail from the end of the previous chunk.
            tail, tail_tokens = [], 0
            for u in reversed(current):
                t = _approx_tokens(u)
                if tail_tokens + t > overlap:
                    break
                tail.insert(0, u)
                tail_tokens += t
            current = tail
            current_tokens = tail_tokens
        current.append(unit)
        current_tokens += ut

    flush()
    return chunks