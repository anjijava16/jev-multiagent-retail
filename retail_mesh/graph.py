"""The orchestrator: a LangGraph graph whose decisions all come from Jev.

    START
      -> screen       Jev: guard (3 Nouls) + triage (Choice, 2 Scores, 3 Nouls)
                      = 9 questions in ONE call, evaluated in parallel
      -> [router]     plain Python: block? else thresholds -> Send() to 1..N agents
      -> sub-agents   each is its own LangGraph graph, run in parallel
      -> synthesize   one agent: pass through; several: merge with an LLM
      -> verify       Jev: is the reply grounded in the facts?          1 call
      -> finalize | human_handoff
    END

No LLM is ever asked "what should I do next?". Jev answers narrow typed
questions; Python decides; LLMs only write prose.
"""
from __future__ import annotations

import hashlib

from langgraph.graph import END, START, StateGraph

try:
    from langgraph.types import Send
except ImportError:  # older langgraph
    from langgraph.constants import Send

from . import llm
from .agents import (complaint_agent, order_agent, product_agent, recommend_agent,
                     returns_agent)
from .agents.base import result
from .config import MODELS, TH
from .jev_client import Answer, get_jev
from .questions import grounded_question, guard_questions, triage_questions
from .router import HUMAN, plan_route
from .state import Ticket

SUB_AGENTS = {
    "order_agent": order_agent.GRAPH,
    "product_agent": product_agent.GRAPH,
    "returns_agent": returns_agent.GRAPH,
    "recommend_agent": recommend_agent.GRAPH,
    "complaint_agent": complaint_agent.GRAPH,
}


def _fmt(a: Answer) -> str:
    if a.type == "choice":
        return f"{a.choice}({a.confidence:.2f})"
    if a.type == "score":
        return f"{a.score:.2f}({a.confidence:.2f})"
    return f"{a.noul:.2f}"


# ------------------------------------------------------------------ screen
BLOCK_INJECTION = ("I can't help with that, but I'm happy to help with "
                   "orders, products, returns or recommendations.")
BLOCK_OFF_TOPIC = ("I'm the ShopSense shopping assistant, so I can only help "
                   "with orders, products, returns and gift ideas.")


def screen(t: Ticket) -> dict:
    """Guard + triage in ONE Jev request: 3 guard Nouls + 6 triage questions.

    Jev answers all 9 in parallel. Python then looks at the guard answers first;
    the triage answers are only used if the message passes the guard.
    """
    g_q, t_q = guard_questions(), triage_questions()
    try:
        a = get_jev().ask({"message": t["message"]}, {**g_q, **t_q}, tag="screen")
    except Exception as exc:  # Jev unreachable: never guess, hand to a person
        return {"plan": [HUMAN], "plan_reason": f"jev unavailable ({type(exc).__name__})",
                "trace": [f"screen: Jev call failed ({exc}); routing to human"]}

    guard = {k: a[k] for k in g_q}
    tri = {k: a[k] for k in t_q}
    out = {"guard": {k: v.to_dict() for k, v in guard.items()},
           "triage": {k: v.to_dict() for k, v in tri.items()},
           "trace": ["screen.guard: " + " ".join(f"{k}={_fmt(v)}" for k, v in guard.items()),
                     "screen.triage: " + " ".join(f"{k}={_fmt(v)}" for k, v in tri.items())]}

    if guard["prompt_injection"].noul >= TH.injection:
        out.update(blocked=True, final=BLOCK_INJECTION)
        out["trace"].append("router: blocked (prompt injection)")
    elif guard["off_topic"].noul >= TH.off_topic:
        out.update(blocked=True, final=BLOCK_OFF_TOPIC)
        out["trace"].append("router: blocked (off topic)")
    else:
        plan, reason = plan_route(tri)
        out.update(plan=plan, plan_reason=reason)
        out["trace"].append(f"router: {plan} because {reason}")
    return out


def after_screen(t: Ticket):
    """Blocked -> END. Otherwise one Send per planned agent (run in parallel)."""
    if t.get("blocked"):
        return END
    return [Send(name, t) for name in t["plan"]]


# ------------------------------------------------------------- sub-agents
def make_agent_node(name: str):
    graph = SUB_AGENTS[name]

    def node(t: Ticket) -> dict:
        try:
            out = graph.invoke({"message": t["message"], "customer_id": t["customer_id"],
                                "triage": t.get("triage", {}), "trace": []})
        except Exception as exc:  # Jev or data error inside an agent: hand to a person
            out = {"reply": "", "needs_human": True,
                   "handoff_reason": f"{name} failed ({type(exc).__name__})",
                   "trace": [f"{name}: error {exc!r}"]}
        res = result(name, out)
        return {"agent_outputs": [res],
                "trace": out.get("trace", []) + [f"{name}: reply via {res['reply_source']}"]}

    return node


# ------------------------------------------------------------- synthesize
def synthesize(t: Ticket) -> dict:
    outs = t.get("agent_outputs", [])
    if len(outs) == 1:
        return {"draft": outs[0]["reply"], "trace": ["synthesize: single agent, pass-through"]}
    fallback = "\n\n".join(o["reply"] for o in outs)
    facts = {o["agent"]: {"reply": o["reply"], "facts": o["facts"]} for o in outs}
    text, src = llm.complete(MODELS["synth"],
                             "Merge these agent replies into ONE reply that answers every part "
                             f"of the customer's message: {t['message']}. Keep every fact as is.",
                             facts, fallback)
    return {"draft": text, "trace": [f"synthesize: merged {len(outs)} agent replies via {src}"]}


# ----------------------------------------------------------------- verify
def verify(t: Ticket) -> dict:
    facts = {o["agent"]: o["facts"] for o in t.get("agent_outputs", [])}
    try:
        a = get_jev().ask({"facts": facts, "reply": t["draft"], "message": t["message"]},
                          grounded_question(), tag="verify")
        g = a["grounded"].noul
    except Exception as exc:  # can't verify -> don't send unverified text
        return {"grounded": 0.0, "trace": [f"verify: Jev call failed ({exc})"]}
    return {"grounded": g, "trace": [f"verify: grounded={g:.2f}"]}


def after_verify(t: Ticket) -> str:
    if any(o["needs_human"] for o in t.get("agent_outputs", [])):
        return HUMAN
    return "finalize" if t["grounded"] >= TH.grounded else HUMAN


def finalize(t: Ticket) -> dict:
    return {"final": t["draft"], "trace": ["finalize: sent to customer"]}


# ----------------------------------------------------------- human hand-off
def human_handoff(t: Ticket) -> dict:
    tri = {k: Answer.from_dict(v) for k, v in t.get("triage", {}).items()}
    frustration = tri["frustration"].score if "frustration" in tri else 0
    reasons = [o["handoff_reason"] for o in t.get("agent_outputs", []) if o["needs_human"]]
    if not reasons and t.get("grounded") is not None and t["grounded"] < TH.grounded:
        reasons.append(f"reply failed grounding check ({t['grounded']:.2f})")
    if not reasons:
        reasons.append(t.get("plan_reason", "routed to human"))
    urgent = frustration >= 1.5 or any("safety" in r or "legal" in r for r in reasons)
    ticket_id = "HT-" + hashlib.sha1(t["message"].encode()).hexdigest()[:6].upper()
    handoff = {"ticket": ticket_id, "priority": "P1" if urgent else "P3", "reasons": reasons,
               "draft_for_agent": t.get("draft", "")}
    when = "within the hour" if urgent else "within one business day"
    return {"handoff": handoff,
            "final": f"I've passed this to a member of our support team (ticket {ticket_id}). "
                     f"Someone will reply {when}.",
            "trace": [f"human_handoff: {ticket_id} {handoff['priority']} reasons={reasons}"]}


# ------------------------------------------------------------------ build
def build_graph():
    g = StateGraph(Ticket)
    g.add_node("screen", screen)
    for name in SUB_AGENTS:
        g.add_node(name, make_agent_node(name))
    g.add_node("synthesize", synthesize)
    g.add_node("verify", verify)
    g.add_node("finalize", finalize)
    g.add_node(HUMAN, human_handoff)

    g.add_edge(START, "screen")
    g.add_conditional_edges("screen", after_screen, [*SUB_AGENTS, HUMAN, END])
    for name in SUB_AGENTS:
        g.add_edge(name, "synthesize")
    g.add_edge("synthesize", "verify")
    g.add_conditional_edges("verify", after_verify, ["finalize", HUMAN])
    g.add_edge("finalize", END)
    g.add_edge(HUMAN, END)
    return g.compile()


APP = build_graph()


def run(message: str, customer_id: str = "C-1001") -> Ticket:
    return APP.invoke({"message": message, "customer_id": customer_id,
                       "agent_outputs": [], "trace": []})
