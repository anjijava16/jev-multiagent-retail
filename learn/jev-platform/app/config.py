"""All runtime configuration lives here. Everything is overridable via env / .env."""
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "JEV Platform"
    log_level: str = "INFO"

    # ---- LLM provider ---------------------------------------------------------
    # "openai" or "anthropic". Worker = does the actual work, router = pre-routing
    # classifier, judge = post-check. Keep router/judge cheap; worker can be big.
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    openai_worker_model: str = "gpt-4.1"
    openai_router_model: str = "gpt-4.1-mini"
    openai_judge_model: str = "gpt-4.1-mini"

    anthropic_worker_model: str = "claude-sonnet-5-5"
    anthropic_router_model: str = "claude-haiku-4-5-20251001"
    anthropic_judge_model: str = "claude-haiku-4-5-20251001"

    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2

    # ---- Embeddings / vector store -------------------------------------------
    # Anthropic has no embedding endpoint, so embeddings stay on OpenAI even when
    # the chat provider is Claude. "hash" is an offline dev/test embedding.
    embedding_provider: Literal["openai", "hash"] = "openai"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    vector_backend: Literal["opensearch", "memory"] = "opensearch"
    opensearch_url: str = "http://localhost:9200"
    opensearch_user: str | None = None
    opensearch_password: str | None = None
    opensearch_verify_certs: bool = False
    opensearch_index: str = "jev-knowledge"

    chunk_size: int = 900          # characters
    chunk_overlap: int = 150
    retrieval_k: int = 6
    retrieval_candidates: int = 30  # per leg (BM25 / kNN) before RRF fusion
    rrf_k: int = 60
    min_retrieval_score: float = 0.01  # RRF scores are small; this filters noise

    # ---- JEV policy -----------------------------------------------------------
    max_attempts: int = 3                  # total executions incl. the first
    accept_threshold: float = 0.72         # aggregate post-check score to accept
    judge_gray_zone: tuple[float, float] = (0.55, 0.75)  # escalate judge tier inside this band
    enable_judge_escalation: bool = True   # second opinion from the worker model in gray zone
    risk_escalation_threshold: float = 0.8 # pre-route risk above this goes to a human
    router_llm_weight: float = 0.7         # blend LLM classifier vs heuristics

    agent_max_steps: int = 5
    tool_max_iterations: int = 6

    trace_store_size: int = 500


@lru_cache
def get_settings() -> Settings:
    return Settings()
