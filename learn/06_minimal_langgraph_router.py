"""Lesson 6: the smallest possible "Jev as orchestrator" in LangGraph.

    START -> classify (Jev) --conditional edge--> order | returns | human -> END

This is the whole idea of the project in ~50 lines. The full version in
retail_mesh/graph.py adds guardrails, parallel sub-agents, LLMs and a
grounding check, but the shape is the same.
"""
from typing import TypedDict

from _setup import get_jev
from langgraph.graph import END, START, StateGraph


class S(TypedDict, total=False):
    message: str
    intent: str
    confidence: float
    reply: str


def classify(s: S) -> dict:
    a = get_jev().ask({"message": s["message"]}, {"intent": {
        "type": "choice", "instructions": "The customer's main request",
        "criteria": {"order_status": "Where is an order", "return_refund": "Return or refund",
                     "other": "Anything else"}}})["intent"]
    print(a)
    return {"intent": a.choice, "confidence": a.confidence}


def route(s: S) -> str:                     # plain Python decides, using Jev's numbers
    if s["confidence"] < 0.5:
        return "human"
    return {"order_status": "order", "return_refund": "returns"}.get(s["intent"], "human")


def order(s: S) -> dict:
    return {"reply": "Order agent: looking that up for you."}


def returns(s: S) -> dict:
    return {"reply": "Returns agent: let's start your return."}


def human(s: S) -> dict:
    return {"reply": "Handing you to a person."}


g = StateGraph(S)
for name, fn in [("classify", classify), ("order", order), ("returns", returns), ("human", human)]:
    g.add_node(name, fn)
g.add_edge(START, "classify")
g.add_conditional_edges("classify", route, ["order", "returns", "human"])
for n in ("order", "returns", "human"):
    g.add_edge(n, END)
app = g.compile()

for m in [
        #"Where is SS-10421?"
           "I want to return my jacket", "asdf"
          ]:
    out = app.invoke({"message": m})
    print(f"{m:<30} intent={out['intent']:<14} conf={out['confidence']:.2f} -> {out['reply']}")
