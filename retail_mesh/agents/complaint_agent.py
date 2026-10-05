"""Complaint agent.

assess  -> Jev: severity (Score 0-3), safety (Noul), legal threat (Noul), wants compensation (Noul)
decide  -> Python: safety/legal -> human, else credit from a severity table
respond -> LLM (Claude by default) writes the apology using only the decided facts
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .. import llm
from ..config import MODELS, TH
from ..data import COMPLAINT_CREDIT_BY_SEVERITY, CUSTOMERS
from ..jev_client import get_jev
from ..questions import complaint_questions
from .base import AgentState, latest_order, money


def assess(s: AgentState) -> dict:
    a = get_jev().ask({"message": s["message"]}, complaint_questions(), tag="complaint.assess")
    f = {"severity": a["severity"].score, "severity_conf": a["severity"].confidence,
         "safety_p": a["safety_issue"].noul, "legal_p": a["legal_threat"].noul,
         "wants_comp_p": a["wants_compensation"].noul}
    return {"facts": f, "trace": [
        f"complaint_agent: severity={f['severity']:.2f} (conf {f['severity_conf']:.2f}) "
        f"safety={f['safety_p']:.2f} legal={f['legal_p']:.2f}"]}


def decide(s: AgentState) -> dict:
    f = dict(s["facts"])
    if f["safety_p"] >= TH.safety:
        return {"facts": f, "needs_human": True, "handoff_reason": "possible safety incident",
                "trace": ["complaint_agent: safety -> human, no automated offer"]}
    if f["legal_p"] >= TH.legal:
        return {"facts": f, "needs_human": True, "handoff_reason": "legal / chargeback threat",
                "trace": ["complaint_agent: legal -> human"]}
    level = round(f["severity"])
    if f["severity_conf"] < 0.3:  # unsure: be generous by one level rather than stingy
        level = min(3, level + 1)
    credit = COMPLAINT_CREDIT_BY_SEVERITY[level]
    if CUSTOMERS.get(s["customer_id"], {}).get("tier") == "gold":
        credit *= 2
    hit = latest_order(s["customer_id"])
    f.update(credit=credit, credit_display=money(credit), order_id=hit[0] if hit else None)
    return {"facts": f, "trace": [f"complaint_agent: severity level {level} -> credit {money(credit)}"]}


def respond(s: AgentState) -> dict:
    f = s["facts"]
    if s.get("needs_human"):
        return {"reply": "I'm really sorry this happened. I'm getting a member of our team "
                         "involved right away.", "reply_source": "template"}
    fallback = "I'm sorry about this experience, that's not the standard we aim for."
    if f["credit"]:
        fallback += f" I've added {f['credit_display']} in store credit to your account."
    fallback += " If you'd like a replacement or a refund as well, just let me know."
    text, src = llm.complete(MODELS["complaint_agent"],
                             f"Apologise sincerely for: {s['message']}. Mention the credit if any.",
                             f, fallback)
    return {"reply": text, "reply_source": src}


def build():
    g = StateGraph(AgentState)
    for name, fn in (("assess", assess), ("decide", decide), ("respond", respond)):
        g.add_node(name, fn)
    g.add_edge(START, "assess")
    g.add_edge("assess", "decide")
    g.add_edge("decide", "respond")
    g.add_edge("respond", END)
    return g.compile()


GRAPH = build()
