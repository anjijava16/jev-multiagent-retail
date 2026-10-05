"""
The whole JEV loop as one LangGraph state machine.

            ┌──────────────┐
  START ──▶ │  pre_route   │ classify · score · prioritize · route
            └──────┬───────┘
     ┌─────────┬───┴─────┬──────────┬───────────┐
     ▼         ▼         ▼          ▼           ▼
 llm_only     rag      tools      agent      escalate ──▶ END
     └─────────┴────┬────┴──────────┘           ▲
                    ▼                           │
            ┌──────────────┐   escalate         │
            │  post_check  │────────────────────┘
            └──────┬───────┘
          accept   │   retry / re-route (back to one of the 4 executors)
            ▼
         finalize ──▶ END

Retries are edges back into the executor nodes, so the full path a request took
(e.g. llm_only -> rag -> rag[expand]) is visible in `history` and in the trace.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.jev.escalation import EscalationQueue
from app.jev.models import Action, ExecutionResult, Route, RouteDecision, Verdict
from app.jev.post_checker import PostChecker
from app.jev.pre_router import PreRouter
from app.observability.tracing import span


class JEVState(TypedDict, total=False):
    trace_id: str
    query: str
    user_id: str | None
    filters: dict | None
    force_route: Route | None
    max_attempts: int | None

    decision: RouteDecision
    route: Route
    strategy: dict[str, Any]
    attempt: int
    result: ExecutionResult
    verdict: Verdict
    history: Annotated[list[dict[str, Any]], operator.add]

    status: str              # answered | escalated
    final_answer: str
    escalation_id: str | None


EXECUTOR_NODES = {Route.LLM_ONLY: "llm_only", Route.RAG: "rag", Route.TOOLS: "tools", Route.AGENT: "agent"}

ESCALATION_MESSAGE = ("I've passed this to a specialist so you get a reliable answer. "
                      "Reference: {id}. You'll hear back shortly.")


def build_jev_graph(pre_router: PreRouter, post_checker: PostChecker, executors: dict[Route, Any],
                    escalations: EscalationQueue):

    # ------------------------------------------------------------------ nodes
    async def pre_route(st: JEVState) -> dict:
        d = await pre_router.decide(st["query"], st.get("force_route"))
        return {"decision": d, "route": d.route, "strategy": {"filters": st.get("filters")}, "attempt": 0}

    def make_exec_node(route: Route):
        async def node(st: JEVState) -> dict:
            strategy = {"filters": st.get("filters"), **st.get("strategy", {})}
            try:
                result = await executors[route].run(st["query"], st["decision"], strategy)
            except Exception as e:  # executor crash is a failed attempt, not a 500
                result = ExecutionResult(route=route, error=f"{type(e).__name__}: {e}")
            return {"result": result, "attempt": st.get("attempt", 0) + 1}
        node.__name__ = f"exec_{route.value}"
        return node

    async def post_check(st: JEVState) -> dict:
        r = st["result"]
        skey = st.get("strategy", {}).get("strategy_key", "")
        # include the attempt we're judging, so the retry planner never picks it again
        tried = [*st.get("history", []), {"route": r.route.value, "strategy_key": skey}]
        v = await post_checker.check(st["query"], st["decision"], r, st["attempt"], tried,
                                     st.get("max_attempts"))
        entry = {
            "attempt": st["attempt"], "route": r.route.value,
            "strategy_key": skey,
            "score": v.score, "action": v.action.value, "judge_tier": v.judge_tier,
            "failed_checks": [c.name for c in v.checks if not c.passed],
            "latency_ms": r.latency_ms, "reason": v.reason,
        }
        upd: dict[str, Any] = {"verdict": v, "history": [entry]}
        if v.action == Action.RETRY and v.next_route:
            upd["route"] = v.next_route
            upd["strategy"] = v.next_strategy
        return upd

    async def finalize(st: JEVState) -> dict:
        return {"status": "answered", "final_answer": st["result"].answer}

    async def escalate(st: JEVState) -> dict:
        with span("jev.escalate"):
            d: RouteDecision = st["decision"]
            v: Verdict | None = st.get("verdict")
            reason = v.reason if v else d.reason
            res = st.get("result")
            e = escalations.open(trace_id=st["trace_id"], query=st["query"], reason=reason,
                                 priority=d.priority.value, draft_answer=res.answer if res else None,
                                 route_history=st.get("history", []))
            return {"status": "escalated", "escalation_id": e.id,
                    "final_answer": ESCALATION_MESSAGE.format(id=e.id)}

    # ------------------------------------------------------------------ edges
    def after_pre_route(st: JEVState) -> str:
        return "escalate" if st["route"] == Route.ESCALATE else EXECUTOR_NODES[st["route"]]

    def after_post_check(st: JEVState) -> str:
        v = st["verdict"]
        if v.action == Action.ACCEPT:
            return "finalize"
        if v.action == Action.RETRY and v.next_route in EXECUTOR_NODES:
            return EXECUTOR_NODES[v.next_route]
        return "escalate"

    g = StateGraph(JEVState)
    g.add_node("pre_route", pre_route)
    for route, name in EXECUTOR_NODES.items():
        g.add_node(name, make_exec_node(route))
        g.add_edge(name, "post_check")
    g.add_node("post_check", post_check)
    g.add_node("finalize", finalize)
    g.add_node("escalate", escalate)

    g.add_edge(START, "pre_route")
    exec_targets = {n: n for n in EXECUTOR_NODES.values()}
    g.add_conditional_edges("pre_route", after_pre_route, {**exec_targets, "escalate": "escalate"})
    g.add_conditional_edges("post_check", after_post_check,
                            {**exec_targets, "finalize": "finalize", "escalate": "escalate"})
    g.add_edge("finalize", END)
    g.add_edge("escalate", END)
    return g.compile()
