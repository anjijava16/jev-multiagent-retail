from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.jev.models import Citation, Route, ToolCallRecord


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000, examples=["What is the return window for electronics?"])
    user_id: str | None = Field(default=None, examples=["u-123"])
    filters: dict[str, Any] | None = Field(default=None, description="Metadata filters for RAG, e.g. {'department': 'support'}")
    force_route: Route | None = Field(default=None, description="Skip pre-routing (debugging / evals). Policy overrides still apply.")
    max_attempts: int | None = Field(default=None, ge=1, le=5)
    debug: bool = Field(default=False, description="Include full pre-route and verdict details")


class AttemptSummary(BaseModel):
    attempt: int
    route: str
    strategy_key: str = ""
    score: float
    action: str
    judge_tier: int
    failed_checks: list[str]
    latency_ms: float
    reason: str


class AskResponse(BaseModel):
    trace_id: str
    status: str                       # answered | escalated
    answer: str
    route_taken: str | None
    initial_route: str
    intent: str
    priority: str
    attempts: list[AttemptSummary]
    final_score: float | None
    citations: list[Citation] = []
    tool_calls: list[ToolCallRecord] = []
    plan: list[str] = []
    escalation_id: str | None = None
    latency_ms: float
    debug: dict[str, Any] | None = None


class IngestDoc(BaseModel):
    text: str = Field(min_length=1)
    source: str = Field(examples=["policies/returns.md"])
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[IngestDoc]


class IngestResponse(BaseModel):
    documents: int
    chunks: int


class RetrieveRequest(BaseModel):
    query: str
    k: int = 5
    filters: dict[str, Any] | None = None
    expand: bool = False


class ResolveRequest(BaseModel):
    resolution: str
