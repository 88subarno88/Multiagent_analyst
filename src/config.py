"""
Central configuration.

Everything tunable lives here so experiments are reproducible: when you write a
results file you can also record which settings produced it. Reads from a .env
file (see .env.example) but every field has a sane default so the app still
imports without one.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- API keys (set in .env) ---
    gemini_api_key: str = ""
    tavily_api_key: str = ""

    # --- Database ---
    # Matches the docker-compose Postgres+pgvector service below.
    database_url: str = "postgresql://research:research@localhost:5432/research"

    # --- Model registry ---
    # Provider-agnostic: switch primary provider with one flag to get the
    # cost/quality tradeoff row in your benchmark table (Milestone 5).
    primary_provider: str = "gemini"          # "gemini" | "ollama"
    gemini_model: str = "gemini-2.5-flash"     # generous free tier
    ollama_model: str = "llama3.1:8b"          # open-weight, local, free
    ollama_base_url: str = "http://localhost:11434"

    # Local embedding model. all-MiniLM-L6-v2 -> 384 dims (matches schema.sql).
    # If you change this, update vector(N) in schema.sql.
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Cross-encoder reranker (local, free). Lazy-loaded; if download fails the
    # pipeline degrades gracefully and skips reranking.
    reranker_model: str = "BAAI/bge-reranker-base"

    # --- Retrieval / RAG knobs ---
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 50
    retrieval_k: int = 5            # final chunks fed to the synthesizer
    overfetch_k: int = 20          # candidates pulled before reranking
    rrf_k: int = 60                # Reciprocal Rank Fusion constant
    memory_hit_threshold: float = 0.78  # cosine sim above which we reuse memory

    # --- Agent / orchestration ---
    max_subquestions: int = 6
    max_corrective_retries: int = 1   # corrective-RAG retry budget per worker

    # --- Cost & latency budget (Milestone 4) ---
    max_tokens_per_query: int = 120_000
    enable_cache: bool = True

    # --- Observability (optional; no-ops if blank) ---
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # Rough $/1M-token prices for cost accounting (update as needed).
    # Gemini 2.5 Flash free tier ~ $0, but we price it so the math is portable.
    price_in_per_mtok: float = 0.30
    price_out_per_mtok: float = 2.50


settings = Settings()