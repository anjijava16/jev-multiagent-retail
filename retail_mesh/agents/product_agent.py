"""Product Q&A agent.

retrieve (code, keyword match) -> rerank (Jev Choice over SKUs) -> respond (LLM)

The Jev rerank is the "line-by-line search" pattern from the TypeSafe cookbook:
put the candidates in the state, ask ONE Choice whose options are the SKUs, and
read the probability of every option as a relevance score.
"""
from __future__ import annotations

import re

from langgraph.graph import END, START, StateGraph

from .. import llm
from ..config import MODELS
from ..data import CATALOG
from ..jev_client import get_jev
from ..questions import product_rerank_questions
from .base import AgentState, budget_in, money

_W = re.compile(r"[a-z0-9\-]+")


def _toks(t: str) -> set[str]:
    return {w.rstrip("s") for w in _W.findall(t.lower()) if len(w) > 2}


def retrieve(s: AgentState) -> dict:
    q = _toks(s["message"])
    budget = budget_in(s["message"])   # "under $100" is a hard filter, so it's code
    scored = []
    for p in CATALOG:
        if budget and p["price"] > budget:
            continue
        text = " ".join([p["name"], p["description"], " ".join(p["tags"]), p["category"]])
        overlap = len(q & _toks(text))
        if overlap:
            scored.append((overlap, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    cands = [p for _, p in scored[:6]]
    return {"facts": {"candidates": cands},
            "trace": [f"product_agent: keyword retrieval (budget={budget}) -> "
                      f"{[p['sku'] for p in cands]}"]}


def rerank(s: AgentState) -> dict:
    cands = s["facts"]["candidates"]
    if not cands:
        return {"facts": {"candidates": [], "ranked": [], "has_match": 0.0},
                "trace": ["product_agent: nothing retrieved, skipping Jev"]}
    state = {
        "message": s["message"],
        "products": {p["sku"]: {k: p[k] for k in ("name", "description", "price", "in_stock")}
                     for p in cands},
    }
    a = get_jev().ask(state, product_rerank_questions(cands), tag="product.rerank")
    best = a["best_match"]
    by_sku = {p["sku"]: p for p in cands}
    ranked = [dict(by_sku[sku], relevance=round(p, 3)) for sku, p in best.top(3)]
    has = a["catalog_has_match"].noul
    return {"facts": {"candidates": cands, "ranked": ranked, "has_match": has},
            "trace": [f"product_agent: Jev best={best.choice} conf={best.confidence:.2f} "
                      f"has_match={has:.2f}"]}


def respond(s: AgentState) -> dict:
    f = s["facts"]
    if f.get("has_match", 0) < 0.4 or not f.get("ranked"):
        facts = {"match": False}
        fallback = "I couldn't find anything in our catalog that matches that. Could you tell me a bit more?"
    else:
        top = [p for p in f["ranked"] if p["relevance"] >= 0.1][:3]
        facts = {"match": True, "products": [
            {"sku": p["sku"], "name": p["name"], "price": p["price"],
             "price_display": money(p["price"]), "in_stock": p["in_stock"],
             "description": p["description"]} for p in top]}
        lead = top[0]
        stock = "in stock" if lead["in_stock"] else "currently out of stock"
        fallback = (f"Yes, we carry the {lead['name']} ({money(lead['price'])}, {stock}). "
                    f"{lead['description']}")
        if len(top) > 1:
            fallback += " You might also like: " + ", ".join(
                f"{p['name']} ({money(p['price'])})" for p in top[1:]) + "."
    text, src = llm.complete(MODELS["product_agent"],
                             f"Answer the customer's product question: {s['message']}",
                             facts, fallback)
    return {"facts": facts, "reply": text, "reply_source": src}


def build():
    g = StateGraph(AgentState)
    for name, fn in (("retrieve", retrieve), ("rerank", rerank), ("respond", respond)):
        g.add_node(name, fn)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "respond")
    g.add_edge("respond", END)
    return g.compile()


GRAPH = build()
