from __future__ import annotations

import operator
import re
from datetime import date
from typing import Annotated, TypedDict

from ..data import CATALOG_BY_SKU, ORDERS, TODAY, orders_for

ORDER_ID = re.compile(r"\bSS-\d{5}\b", re.I)
BUDGET = re.compile(r"\$\s?(\d+)|(\d+)\s?(?:dollars|usd|bucks)", re.I)


def budget_in(text: str) -> float | None:
    m = BUDGET.search(text)
    return float(m.group(1) or m.group(2)) if m else None


class AgentState(TypedDict, total=False):
    message: str
    customer_id: str
    triage: dict
    facts: dict
    reply: str
    reply_source: str
    needs_human: bool
    handoff_reason: str
    trace: Annotated[list[str], operator.add]


def result(agent: str, s: AgentState) -> dict:
    """What every sub-agent hands back to the orchestrator."""
    return {
        "agent": agent,
        "reply": s.get("reply", ""),
        "reply_source": s.get("reply_source", "template"),
        "facts": s.get("facts", {}),
        "needs_human": s.get("needs_human", False),
        "handoff_reason": s.get("handoff_reason", ""),
    }


def money(x: float) -> str:
    return f"${x:,.2f}"


def nice_date(iso: str | None) -> str:
    return date.fromisoformat(iso).strftime("%b %d, %Y") if iso else "not yet"


def order_ids_in(text: str) -> list[str]:
    return [m.upper() for m in ORDER_ID.findall(text)]


def owned_order(order_id: str, customer_id: str) -> dict | None:
    """Never reveal an order to someone who doesn't own it."""
    o = ORDERS.get(order_id)
    return o if o and o["customer_id"] == customer_id else None


def latest_order(customer_id: str) -> tuple[str, dict] | None:
    mine = orders_for(customer_id)
    if not mine:
        return None
    oid = max(mine, key=lambda k: mine[k]["placed"])
    return oid, mine[oid]


def item_view(item: dict) -> dict:
    p = CATALOG_BY_SKU[item["sku"]]
    return {"sku": p["sku"], "name": p["name"], "category": p["category"],
            "price": item["price"], "price_display": money(item["price"]),
            "final_sale": p["final_sale"]}


def days_since(iso: str) -> int:
    return (TODAY - date.fromisoformat(iso)).days
