"""
Offline test harness. A scripted FakeGateway stands in for OpenAI/Claude so the
full JEV graph (pre-route -> executors -> post-check -> retry/escalate) runs with
no network, no API keys, and an in-memory vector store.
"""
from __future__ import annotations

import asyncio
import re
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.config import Settings
from app.container import build_container
from app.executors.agent_workflow import Plan
from app.jev.models import JudgeScores, RouteClassification
from app.rag.retriever import QueryRewrites

ROOT = Path(__file__).resolve().parents[1]


class FakeGateway:
    def __init__(self):
        self.judge_queue: list[JudgeScores] = []   # push to script judge results
        self.router_fail = False
        self.calls: list[tuple[str, str]] = []

    def model_name(self, role):
        return f"fake-{role}"

    # ---------------------------------------------------------------- structured
    async def structured(self, role, system, user, schema):
        self.calls.append((role, schema.__name__))
        q = user.lower()
        if schema is RouteClassification:
            if self.router_fail:
                raise TimeoutError("router down")
            if "close my account" in q or "wire" in q:
                return RouteClassification(intent="account_change", needs_knowledge=0.1, needs_action=0.9,
                                           complexity=0.2, risk=0.95, reasoning="irreversible")
            if "order" in q and ("policy" in q or "according" in q):
                return RouteClassification(intent="order_plus_policy", needs_knowledge=0.75, needs_action=0.8,
                                           complexity=0.85, risk=0.3, reasoning="lookup then policy")
            if "order" in q or "calculate" in q:
                return RouteClassification(intent="order_status", needs_knowledge=0.1, needs_action=0.9,
                                           complexity=0.1, risk=0.2, reasoning="needs lookup")
            if any(w in q for w in ("return", "warranty", "refund", "shipping", "policy")):
                return RouteClassification(intent="policy_question", needs_knowledge=0.9, needs_action=0.05,
                                           complexity=0.1, risk=0.2, reasoning="kb")
            return RouteClassification(intent="general", needs_knowledge=0.05, needs_action=0.05,
                                       complexity=0.05, risk=0.05, reasoning="general")
        if schema is JudgeScores:
            if self.judge_queue:
                return self.judge_queue.pop(0)
            return JudgeScores(correct=0.92, relevant=0.95, grounded=0.9, safe=True, confidence=0.9)
        if schema is QueryRewrites:
            return QueryRewrites(queries=["return window electronics restocking fee"])
        if schema is Plan:
            oid = (re.search(r"order\s*#?(\d{5})", q) or [None, "10025"])[1]
            return Plan(steps=[f"Look up order {oid} with lookup_order",
                               "Search the knowledge base for order processing time"])
        raise AssertionError(f"unexpected schema {schema}")

    # ---------------------------------------------------------------- complete
    async def complete(self, role, system, user):
        self.calls.append((role, "complete"))
        if "Context:" in user:
            return "Electronics can be returned within 15 days of delivery [1], with a 10% restocking fee if opened [1]."
        if "Step results" in user:
            return "Order 10025 is still processing; per the shipping policy, processing finishes within 1-2 business days."
        return "A mutex allows one owner at a time; a semaphore allows up to N concurrent holders."

    # ---------------------------------------------------------------- tools
    async def invoke_with_tools(self, role, messages, tools):
        self.calls.append((role, "tools"))
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content=f"Based on the tool result: {messages[-1].content[:200]}")
        text = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
        step = text.split("Now do step")[-1].lower()
        m = re.search(r"order\s*#?(\d{5})", step) or re.search(r"order\s*#?(\d{5})", text.lower())
        if "knowledge base" in step:
            call = {"name": "search_knowledge_base", "args": {"query": "order processing time"}}
        elif m:
            call = {"name": "lookup_order", "args": {"order_id": m.group(1)}}
        elif "ticket" in step:
            call = {"name": "create_support_ticket", "args": {"subject": "help", "description": "x"}}
        else:
            return AIMessage(content="nothing to do")
        return AIMessage(content="", tool_calls=[{**call, "id": uuid.uuid4().hex, "type": "tool_call"}])


@pytest.fixture
def settings():
    return Settings(_env_file=None, vector_backend="memory", embedding_provider="hash", max_attempts=3)


@pytest.fixture
def fake_gw():
    return FakeGateway()


@pytest.fixture
def container(settings, fake_gw):
    c = build_container(settings, gateway=fake_gw)

    async def load():
        for p in sorted((ROOT / "sample_data").glob("*.md")):
            await c.ingest.ingest_text(p.read_text(), p.name, metadata={"department": "support"})
    asyncio.run(load())
    return c


@pytest.fixture
def client(container):
    from app.main import app
    app.state.container = container
    with TestClient(app) as tc:
        yield tc
    app.state.container = None
