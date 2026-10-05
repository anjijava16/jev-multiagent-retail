"""Returns & refunds agent.

locate  -> which order + item? (regex first, Jev Choice if the customer just describes it)
assess  -> Jev: reason (Choice), wanted resolution (Choice), opened? (Noul)
decide  -> Python: dates, return windows, final sale, money. Jev never does arithmetic.
respond -> LLM (Claude by default) writes the reply from the decision.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .. import llm
from ..config import MODELS, TH
from ..data import ORDERS, RETURN_POLICY, orders_for
from ..jev_client import get_jev
from ..questions import item_pick_question, return_assessment_questions
from .base import AgentState, days_since, item_view, nice_date, order_ids_in, owned_order

FAULT_REASONS = {"defective", "damaged_in_shipping", "wrong_item"}


def check_eligibility(item: dict, days: int, reason: str, opened_p: float) -> dict:
    """Pure policy maths. Unit-tested in tests/test_policy.py."""
    if reason in FAULT_REASONS:
        window, label = RETURN_POLICY["defect_window_days"], "fault/damage window"
    elif item["category"] == "electronics" and opened_p >= 0.6:
        window, label = RETURN_POLICY["opened_electronics_window_days"], "opened electronics window"
    else:
        window, label = RETURN_POLICY["standard_window_days"], "standard window"
    if item["final_sale"] and reason not in FAULT_REASONS:
        return {"eligible": False, "why": "final-sale item", "window_days": window,
                "days_since_delivery": days}
    ok = days <= window
    return {"eligible": ok, "why": f"{days} days since delivery, {label} is {window} days",
            "window_days": window, "days_since_delivery": days,
            "free_return_shipping": reason in FAULT_REASONS}


def locate(s: AgentState) -> dict:
    cid, msg = s["customer_id"], s["message"]
    for oid in order_ids_in(msg):
        order = owned_order(oid, cid)
        if order and order["delivered"]:
            items = order["items"]
            if len(items) == 1:
                return {"facts": {"order_id": oid, "item": item_view(items[0]),
                                  "delivered": order["delivered"]},
                        "trace": [f"returns_agent: {oid} has one item"]}
            break  # several items: let Jev pick below, limited to this order
    # Build options from delivered orders and let Jev pick the item being described
    pool = {oid: o for oid, o in orders_for(cid).items() if o["delivered"]}
    ids = order_ids_in(msg)
    if ids and ids[0] in pool:
        pool = {ids[0]: pool[ids[0]]}
    options = {}
    for oid, o in pool.items():
        for it in o["items"]:
            v = item_view(it)
            options[f"{oid}__{it['sku']}"] = f"{v['name']} (order {oid}, delivered {o['delivered']})"
    if not options:
        return {"facts": {"error": "no delivered orders"}, "needs_human": True,
                "handoff_reason": "return request but no delivered orders found",
                "trace": ["returns_agent: no delivered orders"]}
    a = get_jev().ask({"message": msg, "purchases": options}, item_pick_question(options),
                      tag="returns.item")["item"]
    oid, sku = a.choice.split("__")
    item = next(i for i in ORDERS[oid]["items"] if i["sku"] == sku)
    trace = [f"returns_agent: Jev picked {sku} from {oid} (conf {a.confidence:.2f})"]
    facts = {"order_id": oid, "item": item_view(item), "delivered": ORDERS[oid]["delivered"],
             "item_pick_confidence": a.confidence}
    if a.confidence < 0.4 and len(options) > 1:
        return {"facts": facts, "needs_human": True,
                "handoff_reason": "couldn't tell which item the customer means", "trace": trace}
    return {"facts": facts, "trace": trace}


def assess(s: AgentState) -> dict:
    if s.get("needs_human"):
        return {}
    a = get_jev().ask({"message": s["message"], "item": s["facts"]["item"]},
                      return_assessment_questions(), tag="returns.assess")
    f = dict(s["facts"])
    f.update(reason=a["reason"].choice, reason_conf=a["reason"].confidence,
             resolution=a["resolution"].choice, resolution_conf=a["resolution"].confidence,
             opened_p=a["item_opened"].noul)
    return {"facts": f, "trace": [
        f"returns_agent: reason={f['reason']} ({f['reason_conf']:.2f}) "
        f"wants={f['resolution']} ({f['resolution_conf']:.2f}) opened={f['opened_p']:.2f}"]}


def decide(s: AgentState) -> dict:
    if s.get("needs_human"):
        return {}
    f = dict(s["facts"])
    days = days_since(f["delivered"])
    elig = check_eligibility(f["item"], days, f["reason"], f["opened_p"])
    f["decision"] = elig
    f["delivered_display"] = nice_date(f["delivered"])
    f["policy"] = RETURN_POLICY["text"]
    out = {"facts": f, "trace": [f"returns_agent: eligible={elig['eligible']} ({elig['why']})"]}
    # big refunds need a confident read of what the customer wants
    if (elig["eligible"] and f["item"]["price"] >= TH.high_value_refund
            and f["resolution_conf"] < TH.refund_auto_conf):
        out.update(needs_human=True, handoff_reason="high-value refund, intent not clear enough")
    return out


def respond(s: AgentState) -> dict:
    f = s["facts"]
    if "decision" not in f:
        return {"reply": "Let me get a teammate to look at this return with you.", "reply_source": "template"}
    item, d = f["item"], f["decision"]
    want = {"refund": "a refund", "exchange": "an exchange", "store_credit": "store credit"}[f["resolution"]]
    if d["eligible"]:
        fallback = (f"Sorry about the {item['name']}. It's within our return window "
                    f"(delivered {f['delivered_display']}), so I've started {want} "
                    f"of {item['price_display']} for order {f['order_id']}. ")
        fallback += ("Return shipping is on us; a prepaid label is on its way by email."
                     if d.get("free_return_shipping") else
                     "You'll get return instructions by email.")
    elif d["why"] == "final-sale item":
        fallback = (f"The {item['name']} was a final-sale item, so it can't be returned unless "
                    "it's faulty. If something is wrong with it, tell me what happened and I'll take another look.")
    else:
        fallback = (f"I'm sorry, the {item['name']} from order {f['order_id']} was delivered on "
                    f"{f['delivered_display']}, which is outside the {d['window_days']}-day return "
                    "window for this kind of return. If the item is faulty, let me know, since "
                    "faults are covered for 90 days.")
    task = (f"Customer message: {s['message']}\nExplain the return decision kindly. "
            "If eligible, confirm next steps. If not, explain why and mention any option the policy allows.")
    text, src = llm.complete(MODELS["returns_agent"], task, f, fallback)
    return {"reply": text, "reply_source": src}


def build():
    g = StateGraph(AgentState)
    for name, fn in (("locate", locate), ("assess", assess), ("decide", decide), ("respond", respond)):
        g.add_node(name, fn)
    g.add_edge(START, "locate")
    g.add_edge("locate", "assess")
    g.add_edge("assess", "decide")
    g.add_edge("decide", "respond")
    g.add_edge("respond", END)
    return g.compile()


GRAPH = build()
