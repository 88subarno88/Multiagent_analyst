"""
Central configuration.

Everything tunable lives here so experiments are reproducible. Reads from a .env
file but every field has a default so the app still imports without one.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # API keys (set in .env)
    gemini_api_key: str = ""
    gemini_api_keys: str = ""   # comma-separated extras for rotation
    tavily_api_key: str = ""

    # Database (matches the docker-compose Postgres+pgvector service)
    database_url: str = "postgresql://research:research@localhost:5432/research"

    # Model registry
    primary_provider: str = "gemini"          # "gemini" | "ollama"
    gemini_model: str = "gemini-2.5-flash"
    ollama_model: str = "qwen2.5:7b"          # fits RTX 4050 6GB, good at JSON
    ollama_base_url: str = "http://localhost:11434"

    # Local embedding model (384 dims -> matches schema.sql vector(384))
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Cross-encoder reranker (local, lazy-loaded; skips if download fails)
    reranker_model: str = "BAAI/bge-reranker-base"

    # RAG knobs
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 50
    retrieval_k: int = 5
    overfetch_k: int = 20
    rrf_k: int = 60
    memory_hit_threshold: float = 0.78

    max_subquestions: int = 6
    max_corrective_retries: int = 1

    # How many agent runs in flight at once during eval
    eval_concurrency: int = 2

    # Cost & latency budget
    max_tokens_per_query: int = 120_000
    enable_cache: bool = True

    # Observability (optional; no-ops if blank)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # $/1M-token prices for cost accounting
    price_in_per_mtok: float = 0.30
    price_out_per_mtok: float = 2.50

    @property
    def gemini_key_list(self) -> list[str]:
        """All Gemini keys to rotate over (extras + the single key)."""
        keys = [k.strip() for k in self.gemini_api_keys.split(",") if k.strip()]
        if self.gemini_api_key and self.gemini_api_key not in keys:
            keys.append(self.gemini_api_key)
        return keys


settings = Settings()