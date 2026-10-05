"""Data contracts shared by the pre-router, executors and post-checker."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Route(str, Enum):
    LLM_ONLY = "llm_only"
    RAG = "rag"
    TOOLS = "tools"
    AGENT = "agent"
    ESCALATE = "escalate"


class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class Action(str, Enum):
    ACCEPT = "accept"
    RETRY = "retry"
    ESCALATE = "escalate"


# --------------------------------------------------------------------------- #
# Pre-routing
# --------------------------------------------------------------------------- #
class Signals(BaseModel):
    """Cheap deterministic signals. Computed before any LLM call."""
    pii_found: list[str] = Field(default_factory=list)
    injection_suspected: bool = False
    action_cues: list[str] = Field(default_factory=list)
    knowledge_cues: list[str] = Field(default_factory=list)
    multi_step_cues: list[str] = Field(default_factory=list)
    has_math: bool = False
    urgency: bool = False
    risky_action: bool = False  # destructive / money-moving verbs


class RouteClassification(BaseModel):
    """What the router LLM returns (structured output)."""
    intent: str = Field(description="Short snake_case label for what the user wants, e.g. policy_question, order_status")
    needs_knowledge: float = Field(ge=0, le=1, description="Needs private/fresh documents from the knowledge base")
    needs_action: float = Field(ge=0, le=1, description="Needs a tool/API call: lookup, calculate, create, update")
    complexity: float = Field(ge=0, le=1, description="Needs several dependent steps to answer")
    risk: float = Field(ge=0, le=1, description="Harm if wrong: money movement, account changes, legal/medical, data exposure")
    reasoning: str = Field(description="One sentence, why")


class RouteDecision(BaseModel):
    route: Route
    intent: str
    priority: Priority
    confidence: float                    # margin-based, 0..1
    scores: dict[str, float]             # per-route score after blending
    risk: float
    signals: Signals
    classifier: RouteClassification | None = None
    classifier_error: str | None = None
    reason: str


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
class Citation(BaseModel):
    id: int
    source: str
    title: str | None = None
    chunk_id: str
    score: float


class ToolCallRecord(BaseModel):
    name: str
    args: dict[str, Any]
    output: str | None = None
    error: str | None = None
    latency_ms: float = 0.0


class ExecutionResult(BaseModel):
    route: Route
    answer: str = ""
    error: str | None = None
    context: str | None = None                  # what the answer should be grounded in
    citations: list[Citation] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    plan: list[str] = Field(default_factory=list)
    signals: dict[str, Any] = Field(default_factory=dict)  # executor-specific health signals
    latency_ms: float = 0.0


# --------------------------------------------------------------------------- #
# Post-check
# --------------------------------------------------------------------------- #
class JudgeScores(BaseModel):
    """What the judge LLM returns (structured output)."""
    correct: float = Field(ge=0, le=1, description="Factually right and actually answers the request")
    relevant: float = Field(ge=0, le=1, description="On-topic, no padding or drift")
    grounded: float = Field(ge=0, le=1, description="Every claim is supported by CONTEXT. Use 1.0 if no context was provided and the answer is general knowledge")
    safe: bool = Field(description="No harmful content, no leaked secrets/PII, no unauthorized actions")
    confidence: float = Field(ge=0, le=1, description="How sure you are in this assessment")
    issues: list[str] = Field(default_factory=list, description="Concrete problems, empty if none")
    suggested_fix: str = Field(
        default="none",
        description="One of: none, retrieve_more, use_tools, decompose, rephrase",
    )


class CheckResult(BaseModel):
    name: str
    passed: bool
    score: float
    detail: str = ""
    hard_fail: bool = False  # a hard fail can't be outvoted by a good judge score


class Verdict(BaseModel):
    action: Action
    score: float
    checks: list[CheckResult]
    judge: JudgeScores | None = None
    judge_tier: int = 0          # 0 = rules only, 1 = cheap judge, 2 = escalated judge
    next_route: Route | None = None
    next_strategy: dict[str, Any] = Field(default_factory=dict)
    reason: str
