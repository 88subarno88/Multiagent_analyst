"""
Retrieval strategies — this module is the heart of your benchmark table.

Three levels you can toggle to produce the v2/v3/v4 results:
  - dense_only:  vector search alone (the naive baseline most peers stop at)
  - hybrid:      dense + keyword fused with Reciprocal Rank Fusion (RRF)
  - rerank:      hybrid over-fetch, then a cross-encoder reranks the shortlist

Why each matters (interview gold):
  - Dense (bi-encoder) embeds query and chunk separately -> fast, catches
    paraphrase/meaning, but misses exact IDs, rare names, code tokens.
  - Sparse keyword (BM25-ish) nails those exact terms but misses synonyms.
  - RRF fuses two ranked lists without needing comparable scores.
  - A cross-encoder reads (query, chunk) *together* -> far more accurate
    relevance, but too slow to run over the whole corpus, so we only rerank the
    over-fetched shortlist. Bi-encoder for recall, cross-encoder for precision.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import lru_cache

from src.config import settings
from src.memory.store import MemoryStore, Retrieved
from src.models.embeddings import EmbeddingClient


@dataclass
class RetrievalResult:
    chunks: list[Retrieved]
    strategy: str


def _rrf_fuse(*ranked_lists: list[Retrieved], k: int = 60) -> list[Retrieved]:
    """Reciprocal Rank Fusion: score = sum 1/(k + rank). Higher = better."""
    scores: dict[int, float] = {}
    best: dict[int, Retrieved] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            best.setdefault(item.chunk_id, item)
    fused = sorted(best.values(), key=lambda c: scores[c.chunk_id], reverse=True)
    for c in fused:
        c.score = scores[c.chunk_id]
    return fused


@lru_cache(maxsize=1)
def _load_reranker(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


class Retriever:
    def __init__(self, store: MemoryStore, embedder: EmbeddingClient):
        self.store = store
        self.embedder = embedder

    async def retrieve(self, query: str, strategy: str = "rerank") -> RetrievalResult:
        if strategy == "dense_only":
            qvec = await self.embedder.embed_one_async(query)
            chunks = await self.store.dense_search(qvec, settings.retrieval_k)
            return RetrievalResult(chunks, "dense_only")

        # hybrid + rerank both start by over-fetching from both arms in parallel.
        qvec = await self.embedder.embed_one_async(query)
        dense, sparse = await asyncio.gather(
            self.store.dense_search(qvec, settings.overfetch_k),
            self.store.keyword_search(query, settings.overfetch_k),
        )
        fused = _rrf_fuse(dense, sparse, k=settings.rrf_k)

        if strategy == "hybrid":
            return RetrievalResult(fused[: settings.retrieval_k], "hybrid")

        # strategy == "rerank"
        reranked = await self._rerank(query, fused[: settings.overfetch_k])
        return RetrievalResult(reranked[: settings.retrieval_k], "rerank")

    async def _rerank(self, query: str, candidates: list[Retrieved]) -> list[Retrieved]:
        if not candidates:
            return candidates
        try:
            model = _load_reranker(settings.reranker_model)
        except Exception:
            # Graceful degradation: if the reranker can't load, keep fused order.
            return candidates
        pairs = [(query, c.content) for c in candidates]
        scores = await asyncio.to_thread(model.predict, pairs)
        for c, s in zip(candidates, scores):
            c.score = float(s)
        return sorted(candidates, key=lambda c: c.score, reverse=True)