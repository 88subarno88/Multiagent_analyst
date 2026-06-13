"""
Embedding client.

Local sentence-transformers model = zero cost, no rate limits, deterministic.
normalize embeddings so cosine similarity == dot product, which keeps the
pgvector query simple. Embedding is CPU-bound, so we offload to a thread to stay
async-friendly inside the orchestrator.
"""
from __future__ import annotations

import asyncio
from functools import lru_cache

from src.config import settings


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    # Imported lazily so the rest of the app imports fast and you only pay the
    # torch/transformers import cost when you actually embed something.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class EmbeddingClient:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self._model = _load_model(self.model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return [v.tolist() for v in vecs]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed, texts)

    async def embed_one_async(self, text: str) -> list[float]:
        return (await self.embed_async([text]))[0]