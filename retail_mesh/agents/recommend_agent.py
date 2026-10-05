"""Recommendation / gift agent: the "composite scoring" pattern.

candidates (code: budget filter) -> score (ONE Jev call with one Score per product)
-> rank (code: weighted formula you control) -> respond (LLM, Gemini by default)

If marketing wants in-stock items pushed harder, change W_STOCK. No prompt edits.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .. import llm
from ..config import MODELS
from ..data import CATALOG
from ..jev_client import get_jev
from ..questions import fit_questions
from .base import AgentState, budget_in, money

W_FIT, W_STOCK, W_CONF = 0.7, 0.2, 0.1   # composite weights (sum to 1)


def candidates(s: AgentState) -> dict:
    budget = budget_in(s["message"])
    cap = budget * 1.15 if budget else None   # allow a little stretch
    cands = [p for p in CATALOG if not p["final_sale"] and (cap is None or p["price"] <= cap)]
    return {"facts": {"budget": budget, "candidates": cands},
            "trace": [f"recommend_agent: budget={budget} -> {len(cands)} candidates"]}


def score(s: AgentState) -> dict:
    cands = s["facts"]["candidates"]
    if not cands:
        return {"trace": ["recommend_agent: no candidates in budget"]}
    state = {"customer_request": s["message"],
             "candidates": {p["sku"]: {k: p[k] for k in ("name", "description", "tags", "price")}
                            for p in cands}}
    a = get_jev().ask(state, fit_questions(cands), tag=f"recommend.fit x{len(cands)}")
    ranked = []
    for p in cands:
        ans = a[f"fit_{p['sku']}"]
        fit = (ans.score or 0) / 3            # normalise 0..3 -> 0..1
        composite = W_FIT * fit + W_STOCK * p["in_stock"] + W_CONF * (ans.confidence or 0)
        ranked.append({**p, "fit": round(ans.score, 2), "fit_conf": ans.confidence,
                       "composite": round(composite, 3)})
    ranked.sort(key=lambda p: p["composite"], reverse=True)
    f = dict(s["facts"], ranked=ranked[:3])
    return {"facts": f, "trace": [f"recommend_agent: {len(cands)} Score questions in 1 Jev call; "
                                  f"top={[(p['sku'], p['composite']) for p in ranked[:3]]}"]}


def respond(s: AgentState) -> dict:
    f = s["facts"]
    picks = [p for p in f.get("ranked", []) if p["fit"] >= 1.5]
    facts = {"budget": f.get("budget"), "picks": [
        {"sku": p["sku"], "name": p["name"], "price": p["price"], "price_display": money(p["price"]),
         "in_stock": p["in_stock"], "description": p["description"]} for p in picks]}
    if not picks:
        fallback = "I don't have a strong match for that yet. Who is it for, and what do they enjoy?"
    else:
        lines = [f"{p['name']} ({money(p['price'])}): {p['description']}" for p in picks]
        fallback = "Here are a few ideas: " + " | ".join(lines)
    text, src = llm.complete(MODELS["recommend_agent"],
                             f"Recommend products for: {s['message']}. Short reason for each.",
                             facts, fallback)
    return {"facts": facts, "reply": text, "reply_source": src}


def build():
    g = StateGraph(AgentState)
    for name, fn in (("candidates", candidates), ("score", score), ("respond", respond)):
        g.add_node(name, fn)
    g.add_edge(START, "candidates")
    g.add_edge("candidates", "score")
    g.add_edge("score", "respond")
    g.add_edge("respond", END)
    return g.compile()


GRAPH = build()
