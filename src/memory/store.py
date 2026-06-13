"""
Memory store (pgvector on Postgres).

Owns the connection pool and all SQL. Exposes:
  - init_schema()                 run schema.sql once at startup
  - upsert_document(...)          write a scraped doc + its chunks (with embeddings)
  - dense_search(vec, k)          vector cosine nearest neighbours  -> the 'dense' arm
  - keyword_search(query, k)      full-text ranking (~BM25)         -> the 'sparse' arm
  - cache_get / cache_set         repeated-subquestion cache

We register pgvector's asyncpg codec so we can pass/receive Python lists directly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import asyncpg
from pgvector.asyncpg import register_vector

from src.config import settings

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


@dataclass
class Retrieved:
    chunk_id: int
    content: str
    source_url: str | None
    source_title: str | None
    score: float  # dense: cosine similarity; keyword: ts_rank


class MemoryStore:
    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or settings.database_url
        self._pool: asyncpg.Pool | None = None

    async def connect(self):
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self.dsn, min_size=1, max_size=5, init=self._init_conn
            )
        return self._pool

    @staticmethod
    async def _init_conn(conn):
        await register_vector(conn)

    async def close(self):
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def init_schema(self):
        pool = await self.connect()
        async with pool.acquire() as conn:
            await conn.execute(_SCHEMA_PATH.read_text())

    # ---- writes ----
    async def upsert_document(
        self,
        url: str,
        title: str | None,
        content: str,
        chunks: list,           # list[Chunk]
        embeddings: list[list[float]],
    ) -> int:
        pool = await self.connect()
        async with pool.acquire() as conn:
            async with conn.transaction():
                source_id = await conn.fetchval(
                    """INSERT INTO sources (url, title) VALUES ($1, $2)
                       ON CONFLICT (url) DO UPDATE SET title = EXCLUDED.title
                       RETURNING id""",
                    url, title,
                )
                doc_id = await conn.fetchval(
                    "INSERT INTO documents (source_id, content) VALUES ($1, $2) RETURNING id",
                    source_id, content,
                )
                rows = [
                    (doc_id, url, title, c.content, emb, c.token_count)
                    for c, emb in zip(chunks, embeddings)
                ]
                await conn.executemany(
                    """INSERT INTO chunks
                       (document_id, source_url, source_title, content, embedding, token_count)
                       VALUES ($1, $2, $3, $4, $5, $6)""",
                    rows,
                )
        return doc_id

    # ---- reads ----
    async def dense_search(self, query_embedding: list[float], k: int) -> list[Retrieved]:
        pool = await self.connect()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, content, source_url, source_title,
                          1 - (embedding <=> $1) AS score
                   FROM chunks
                   ORDER BY embedding <=> $1
                   LIMIT $2""",
                query_embedding, k,
            )
        return [
            Retrieved(r["id"], r["content"], r["source_url"], r["source_title"], float(r["score"]))
            for r in rows
        ]

    async def keyword_search(self, query: str, k: int) -> list[Retrieved]:
        pool = await self.connect()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, content, source_url, source_title,
                          ts_rank_cd(content_tsv, websearch_to_tsquery('english', $1)) AS score
                   FROM chunks
                   WHERE content_tsv @@ websearch_to_tsquery('english', $1)
                   ORDER BY score DESC
                   LIMIT $2""",
                query, k,
            )
        return [
            Retrieved(r["id"], r["content"], r["source_url"], r["source_title"], float(r["score"]))
            for r in rows
        ]

    # ---- cache ----
    async def cache_get(self, key: str):
        if not settings.enable_cache:
            return None
        pool = await self.connect()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT payload FROM query_cache WHERE cache_key = $1", key)
        return json.loads(row["payload"]) if row else None

    async def cache_set(self, key: str, payload: dict):
        if not settings.enable_cache:
            return
        pool = await self.connect()
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO query_cache (cache_key, payload) VALUES ($1, $2)
                   ON CONFLICT (cache_key) DO UPDATE SET payload = EXCLUDED.payload""",
                key, json.dumps(payload),
            )

    async def count_chunks(self) -> int:
        pool = await self.connect()
        async with pool.acquire() as conn:
            return await conn.fetchval("SELECT count(*) FROM chunks")


# A module-level singleton is convenient for the orchestrator + streamlit app.
store = MemoryStore()