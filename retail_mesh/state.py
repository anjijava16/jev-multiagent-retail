"""LangGraph state shared by the orchestrator."""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class Ticket(TypedDict, total=False):
    # input
    customer_id: str
    message: str
    # guardrails
    guard: dict              # {question_name: Answer.to_dict()}
    blocked: bool
    # triage (Jev)
    triage: dict             # {question_name: Answer.to_dict()}
    plan: list[str]          # agent names chosen by the router
    plan_reason: str
    # sub-agent results; several agents can run in parallel, so lists are merged
    agent_outputs: Annotated[list[dict], operator.add]
    # composing + checking the reply
    draft: str
    grounded: float
    handoff: dict
    final: str
    # human-readable decision log, appended by every node
    trace: Annotated[list[str], operator.add]
