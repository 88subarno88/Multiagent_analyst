-- Memory plane schema (Postgres + pgvector).
-- Run automatically by store.init_schema(); also safe to run by hand with psql.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per web source we've seen.
CREATE TABLE IF NOT EXISTS sources (
    id         SERIAL PRIMARY KEY,
    url        TEXT UNIQUE NOT NULL,
    title      TEXT,
    fetched_at TIMESTAMPTZ DEFAULT now()
);

-- Raw scraped document (before chunking).
CREATE TABLE IF NOT EXISTS documents (
    id         SERIAL PRIMARY KEY,
    source_id  INTEGER REFERENCES sources(id) ON DELETE CASCADE,
    content    TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- The retrieval unit. embedding dim MUST match settings.embedding_dim (384).
CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    source_url  TEXT,
    source_title TEXT,
    content     TEXT NOT NULL,
    embedding   vector(384),
    token_count INTEGER,
    created_at  TIMESTAMPTZ DEFAULT now(),
    -- Generated tsvector powers the sparse (keyword) arm of hybrid search.
    -- Postgres full-text ranking (ts_rank_cd) approximates BM25 closely enough
    -- for this project and keeps everything in one database.
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

-- HNSW index for fast approximate nearest-neighbour cosine search at scale.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- GIN index for the keyword arm.
CREATE INDEX IF NOT EXISTS chunks_tsv_idx
    ON chunks USING GIN (content_tsv);

-- Simple key/value cache for repeated sub-questions (Milestone 4).
CREATE TABLE IF NOT EXISTS query_cache (
    cache_key  TEXT PRIMARY KEY,
    payload    JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);