"""
Agent / Workflow executor: a LangGraph plan -> act -> synthesize subgraph.

    START -> plan -> act --(more steps?)--> act ... -> synthesize -> END

  plan        router-tier model writes 1..N concrete steps (structured output)
  act         worker model executes ONE step with tools (ReAct loop), sees prior step results
  synthesize  worker model writes the final answer from all step results

Why plan-and-execute instead of one big ReAct loop: the plan is inspectable (it's
returned in the API response and the trace), each step has a bounded tool budget,
and the post-checker can judge the final answer against an explicit evidence
trail rather than a long chat transcript.
"""
from __future__ import annotations

import json
import operator
import time
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.config import Settings
from app.executors.basic import _feedback_block, run_tool_loop
from app.jev.models import ExecutionResult, Route, RouteDecision, ToolCallRecord
from app.llm.gateway import LLMGateway
from app.observability.tracing import span


class Plan(BaseModel):
    steps: list[str] = Field(description="Ordered, concrete steps. Each step should need at most one or two tool calls.")


class AgentState(TypedDict):
    query: str
    strategy: dict[str, Any]
    risk: float
    plan: list[str]
    idx: int
    step_results: Annotated[list[str], operator.add]
    tool_calls: Annotated[list[ToolCallRecord], operator.add]
    answer: str


PLAN_SYSTEM = """You plan work for a tool-using agent. Available tools:
{tools}
Write the smallest plan that fully answers the request, max {max_steps} steps.
Don't include a 'write the final answer' step; that happens automatically."""

ACT_SYSTEM = """You are executing ONE step of a plan. Use tools as needed for this step only.
Return a short factual result for this step (what you found or did). Do not answer the whole request."""

SYNTH_SYSTEM = """Write the final answer to the user's request using ONLY the step results.
Be specific (ids, dates, amounts). If a step failed or something is unknown, say so plainly.
If any step result quotes a knowledge-base source, name the source."""


class AgentWorkflowExecutor:
    def __init__(self, gw: LLMGateway, tools: list[BaseTool], s: Settings):
        self.gw, self.tools, self.s = gw, tools, s
        self.graph = self._build()

    # ----------------------------------------------------------------- nodes
    async def _plan(self, st: AgentState) -> dict:
        with span("agent.plan") as a:
            tool_list = "\n".join(f"- {t.name}: {t.description.splitlines()[0]}" for t in self.tools)
            plan = await self.gw.structured(
                "router",
                PLAN_SYSTEM.format(tools=tool_list, max_steps=self.s.agent_max_steps) + _feedback_block(st["strategy"]),
                st["query"], Plan,
            )
            steps = [s for s in plan.steps if s.strip()][: self.s.agent_max_steps] or [st["query"]]
            a["steps"] = len(steps)
            return {"plan": steps, "idx": 0}

    async def _act(self, st: AgentState) -> dict:
        step = st["plan"][st["idx"]]
        with span("agent.act", step=st["idx"] + 1):
            prior = "\n".join(f"Step {i + 1} result: {r}" for i, r in enumerate(st["step_results"])) or "none yet"
            msgs = [
                SystemMessage(ACT_SYSTEM),
                HumanMessage(f"User request: {st['query']}\n\nPlan:\n" +
                             "\n".join(f"{i + 1}. {s}" for i, s in enumerate(st["plan"])) +
                             f"\n\nResults so far:\n{prior}\n\nNow do step {st['idx'] + 1}: {step}"),
            ]
            result, records = await run_tool_loop(self.gw, self.tools, msgs, 4, allow_side_effects=st["risk"] < 0.6)
        return {"idx": st["idx"] + 1,
                "step_results": [result or "step produced no result"],
                "tool_calls": records}

    async def _synthesize(self, st: AgentState) -> dict:
        with span("agent.synthesize"):
            evidence = "\n".join(f"Step {i + 1} ({s}): {r}"
                                 for i, (s, r) in enumerate(zip(st["plan"], st["step_results"])))
            answer = await self.gw.complete("worker", SYNTH_SYSTEM + _feedback_block(st["strategy"]),
                                            f"Request: {st['query']}\n\nStep results:\n{evidence}")
        return {"answer": answer.strip()}

    def _more(self, st: AgentState) -> str:
        return "act" if st["idx"] < len(st["plan"]) else "synthesize"

    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("plan", self._plan)
        g.add_node("act", self._act)
        g.add_node("synthesize", self._synthesize)
        g.add_edge(START, "plan")
        g.add_edge("plan", "act")
        g.add_conditional_edges("act", self._more, {"act": "act", "synthesize": "synthesize"})
        g.add_edge("synthesize", END)
        return g.compile()

    # ----------------------------------------------------------------- entry
    async def run(self, query: str, decision: RouteDecision, strategy: dict[str, Any]) -> ExecutionResult:
        t0 = time.perf_counter()
        with span("exec.agent"):
            out = await self.graph.ainvoke(
                {"query": query, "strategy": strategy, "risk": decision.risk, "plan": [], "idx": 0,
                 "step_results": [], "tool_calls": [], "answer": ""},
                config={"recursion_limit": 4 + 2 * self.s.agent_max_steps},
            )
        records: list[ToolCallRecord] = out["tool_calls"]
        context = "\n".join(
            [f"Step {i + 1}: {r}" for i, r in enumerate(out["step_results"])] +
            [f"{r.name}({json.dumps(r.args)}) -> {r.output or r.error}" for r in records]
        )
        return ExecutionResult(
            route=Route.AGENT, answer=out["answer"], context=context, plan=out["plan"], tool_calls=records,
            signals={"steps": len(out["plan"]), "tool_calls": len(records),
                     "tool_errors": sum(r.error is not None for r in records)},
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
