"""HTTP backend for ShopSense.

    uvicorn api:app --reload --port 8000
    curl -s localhost:8000/v1/chat -H 'content-type: application/json' \
         -d '{"customer_id":"C-1001","message":"Where is my order SS-10421?"}'

The frontend talks only to this endpoint. Jev, the sub-agents and the LLMs are
all behind it; the customer never calls any model directly.
"""
from __future__ import annotations

import time
import uuid

from fastapi import FastAPI
from pydantic import BaseModel, Field

from retail_mesh.graph import run
from retail_mesh.jev_client import get_jev

app = FastAPI(title="ShopSense support API")


class ChatRequest(BaseModel):
    customer_id: str = Field(examples=["C-1001"])
    message: str = Field(examples=["Where is my order SS-10421?"])
    session_id: str | None = None


class Handoff(BaseModel):
    ticket: str
    priority: str
    reasons: list[str]


class ChatResponse(BaseModel):
    request_id: str
    session_id: str
    status: str                 # answered | handed_off | blocked
    reply: str
    route: list[str]            # agents the router picked
    route_reason: str | None
    handoff: Handoff | None
    jev_calls: int
    latency_ms: int
    trace: list[str]            # decision log; hide from end users in production


@app.post("/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    t0, jev = time.perf_counter(), get_jev()
    calls_before = jev.calls
    out = run(req.message, req.customer_id)
    status = "blocked" if out.get("blocked") else "handed_off" if out.get("handoff") else "answered"
    h = out.get("handoff")
    return ChatResponse(
        request_id=f"req_{uuid.uuid4().hex[:10]}",
        session_id=req.session_id or f"sess_{uuid.uuid4().hex[:8]}",
        status=status,
        reply=out.get("final", ""),
        route=out.get("plan", []),
        route_reason=out.get("plan_reason"),
        handoff=Handoff(ticket=h["ticket"], priority=h["priority"], reasons=h["reasons"]) if h else None,
        jev_calls=jev.calls - calls_before,
        latency_ms=int((time.perf_counter() - t0) * 1000),
        trace=out.get("trace", []),
    )


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "jev_mode": get_jev().mode}
