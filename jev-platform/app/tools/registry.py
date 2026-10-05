"""
Tools the tools-executor and the agent can call.

  calculator              safe arithmetic (AST whitelist, no eval)
  current_time            UTC / named timezone
  lookup_order            read-only "API" over a mock order DB
  create_support_ticket   side-effecting write: tagged side_effect=True
  search_knowledge_base   the RAG retriever exposed as a tool, so the agent can
                          decide *when* to read docs instead of always retrieving

The order DB and ticket store are in-memory stand-ins. Swap the function bodies
for real HTTP calls (or an MCP client) and nothing else in the system changes.
"""
from __future__ import annotations

import ast
import operator as op
import uuid
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from langchain_core.tools import BaseTool, StructuredTool

from app.rag.retriever import HybridRetriever

# --------------------------------------------------------------------------- calculator
_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow,
        ast.Mod: op.mod, ast.FloorDiv: op.floordiv, ast.USub: op.neg, ast.UAdd: op.pos}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        if isinstance(node.op, ast.Pow) and abs(_eval(node.right)) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)[:60]}")


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '(1299 * 0.15) + 20'. Supports + - * / // % ** and parentheses."""
    value = _eval(ast.parse(expression.replace("^", "**"), mode="eval").body)
    return str(round(value, 6))


# --------------------------------------------------------------------------- time
def current_time(tz: str = "UTC") -> str:
    """Current date and time in an IANA timezone like 'America/New_York'."""
    return datetime.now(ZoneInfo(tz)).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- orders (mock API)
ORDERS = {
    "10023": {"status": "shipped", "carrier": "UPS", "eta": "2026-10-03", "total": 249.99,
              "items": ["Noise-cancelling headphones"], "category": "electronics", "purchased": "2026-09-12"},
    "10024": {"status": "delivered", "delivered_on": "2026-09-01", "total": 89.50,
              "items": ["Running shoes"], "category": "apparel", "purchased": "2026-08-25"},
    "10025": {"status": "processing", "total": 1299.00, "items": ["Laptop 14in"],
              "category": "electronics", "purchased": "2026-09-28"},
}


def lookup_order(order_id: str) -> str:
    """Look up an order by its numeric id. Returns status, items, totals, dates and category."""
    o = ORDERS.get(order_id.strip().lstrip("#"))
    if not o:
        raise ValueError(f"order {order_id} not found")
    return str({"order_id": order_id, **o})


# --------------------------------------------------------------------------- tickets (side effect)
TICKETS: dict[str, dict] = {}


def create_support_ticket(subject: str, description: str, order_id: str | None = None,
                          priority: str = "normal") -> str:
    """Open a customer-support ticket. Use only when the user asks for one or the issue needs a human."""
    tid = f"TCK-{uuid.uuid4().hex[:6].upper()}"
    TICKETS[tid] = {"subject": subject, "description": description, "order_id": order_id,
                    "priority": priority, "created": datetime.now(timezone.utc).isoformat()}
    return f"created ticket {tid}"


# --------------------------------------------------------------------------- registry
def build_tools(retriever: HybridRetriever) -> list[BaseTool]:
    async def search_knowledge_base(query: str) -> str:
        """Search the company knowledge base (policies, product docs, procedures). Returns numbered passages with sources."""
        hits = await retriever.retrieve(query, k=4)
        if not hits:
            return "no relevant passages found"
        return "\n\n".join(
            f"[{h['source']} :: {h.get('section') or h['title']}]\n{h['text']}" for h in hits
        )

    tools = [
        StructuredTool.from_function(calculator),
        StructuredTool.from_function(current_time),
        StructuredTool.from_function(lookup_order),
        StructuredTool.from_function(create_support_ticket),
        StructuredTool.from_function(coroutine=search_knowledge_base, name="search_knowledge_base",
                                     description=search_knowledge_base.__doc__),
    ]
    for t in tools:
        t.metadata = {"side_effect": t.name == "create_support_ticket"}
        t.handle_tool_error = False  # we catch and record errors ourselves
    return tools
