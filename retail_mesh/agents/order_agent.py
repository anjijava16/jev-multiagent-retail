"""Order status agent. Deliberately has NO LLM and NO Jev call.

Jev already decided this is an order question. Answering it is a database
lookup, so we use code. Cheapest, fastest, can't hallucinate.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .base import (AgentState, item_view, latest_order, nice_date, order_ids_in, owned_order)


def lookup(s: AgentState) -> dict:
    ids = order_ids_in(s["message"])
    trace = []
    if ids:
        oid = ids[0]
        order = owned_order(oid, s["customer_id"])
        if order is None:
            return {"facts": {"order_id": oid, "found": False},
                    "trace": [f"order_agent: {oid} not found for {s['customer_id']}"]}
    else:
        hit = latest_order(s["customer_id"])
        if hit is None:
            return {"facts": {"found": False}, "trace": ["order_agent: customer has no orders"]}
        oid, order = hit
        trace.append(f"order_agent: no order id in message, using most recent ({oid})")
    facts = {
        "found": True, "order_id": oid, "status": order["status"], "carrier": order["carrier"],
        "tracking": order["tracking"], "placed": nice_date(order["placed"]),
        "eta": nice_date(order["eta"]), "delivered": nice_date(order["delivered"]),
        "items": [item_view(i) for i in order["items"]],
    }
    return {"facts": facts, "trace": trace + [f"order_agent: {oid} status={order['status']}"]}


def respond(s: AgentState) -> dict:
    f = s["facts"]
    if not f.get("found"):
        oid = f.get("order_id")
        reply = (f"I couldn't find order {oid} on your account. Could you double-check the number?"
                 if oid else "I couldn't find any orders on your account. Do you have an order number?")
    else:
        names = ", ".join(i["name"] for i in f["items"])
        if f["status"] == "delivered":
            reply = f"Order {f['order_id']} ({names}) was delivered on {f['delivered']}."
        elif f["status"] == "shipped":
            reply = (f"Order {f['order_id']} ({names}) is on its way with {f['carrier']}, "
                     f"tracking {f['tracking']}. Expected delivery: {f['eta']}.")
        else:
            reply = (f"Order {f['order_id']} ({names}) is being prepared and hasn't shipped yet. "
                     f"Current estimate: {f['eta']}.")
    return {"reply": reply, "reply_source": "code"}


def build():
    g = StateGraph(AgentState)
    g.add_node("lookup", lookup)
    g.add_node("respond", respond)
    g.add_edge(START, "lookup")
    g.add_edge("lookup", "respond")
    g.add_edge("respond", END)
    return g.compile()


GRAPH = build()
