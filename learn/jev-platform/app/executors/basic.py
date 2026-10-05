"""
Execution layer, the middle box in the diagram. Each executor takes the request
plus a `strategy` dict that JEV fills in on retries (bigger k, query expansion,
reviewer feedback...) and returns an ExecutionResult with health signals that
the post-checker reads.

Executors never decide whether their own output is good. That's JEV's job.
"""
from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from app.config import Settings
from app.jev.models import Citation, ExecutionResult, Route, RouteDecision, ToolCallRecord
from app.jev.signals import cited_ids
from app.llm.gateway import LLMGateway, message_text
from app.observability.tracing import span
from app.rag.retriever import HybridRetriever


def _feedback_block(strategy: dict[str, Any]) -> str:
    issues = strategy.get("feedback") or []
    if not issues:
        return ""
    return "\n\nA reviewer rejected a previous attempt for these reasons. Fix them:\n- " + "\n- ".join(issues)


# =========================================================================== #
class LLMOnlyExecutor:
    SYSTEM = ("You are a helpful, precise assistant. Answer directly. If the question needs private "
              "company data or live account information you don't have, say so instead of guessing.")

    def __init__(self, gw: LLMGateway):
        self.gw = gw

    async def run(self, query: str, decision: RouteDecision, strategy: dict[str, Any]) -> ExecutionResult:
        t0 = time.perf_counter()
        with span("exec.llm_only"):
            answer = await self.gw.complete("worker", self.SYSTEM + _feedback_block(strategy), query)
        return ExecutionResult(route=Route.LLM_ONLY, answer=answer.strip(),
                               latency_ms=round((time.perf_counter() - t0) * 1000, 1))


# =========================================================================== #
class RAGExecutor:
    SYSTEM = """Answer using ONLY the numbered context passages.
Rules:
- Cite every factual sentence with the passage number in brackets, e.g. [2]. Multiple: [1][3].
- If the context doesn't contain the answer, reply exactly: "I don't have that in the knowledge base."
- Don't mention "the context" or "the passages"; just answer."""

    ABSTAIN = "i don't have that in the knowledge base"

    def __init__(self, gw: LLMGateway, retriever: HybridRetriever, s: Settings):
        self.gw, self.retriever, self.s = gw, retriever, s

    async def run(self, query: str, decision: RouteDecision, strategy: dict[str, Any]) -> ExecutionResult:
        t0 = time.perf_counter()
        k = strategy.get("k", self.s.retrieval_k)
        hits = await self.retriever.retrieve(query, k=k, filters=strategy.get("filters"),
                                             expand=strategy.get("expand", False))
        if not hits:
            return ExecutionResult(route=Route.RAG, signals={"hits": 0, "retrieval_empty": True},
                                   latency_ms=round((time.perf_counter() - t0) * 1000, 1))

        context = "\n\n".join(
            f"[{i}] (source: {h['source']}{' :: ' + h['section'] if h.get('section') else ''})\n{h['text']}"
            for i, h in enumerate(hits, start=1)
        )
        with span("exec.rag.generate", hits=len(hits)):
            answer = (await self.gw.complete(
                "worker", self.SYSTEM + _feedback_block(strategy),
                f"Context:\n{context}\n\nQuestion: {query}",
            )).strip()

        used = cited_ids(answer)
        citations = [Citation(id=i, source=h["source"], title=h.get("title"), chunk_id=h["chunk_id"],
                              score=round(h["score"], 4))
                     for i, h in enumerate(hits, start=1) if i in used]
        return ExecutionResult(
            route=Route.RAG, answer=answer, context=context, citations=citations,
            signals={
                "hits": len(hits),
                "top_score": hits[0]["score"],
                "both_legs_top": hits[0].get("legs", 1) >= 2,   # top doc found by BM25 *and* kNN
                "cited": sorted(used),
                "invalid_citations": sorted(i for i in used if i < 1 or i > len(hits)),
                "abstained": self.ABSTAIN in answer.lower(),
            },
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )


# =========================================================================== #
async def run_tool_loop(gw: LLMGateway, tools: list[BaseTool], messages: list[BaseMessage],
                        max_iter: int, allow_side_effects: bool) -> tuple[str, list[ToolCallRecord]]:
    """Plain ReAct loop: model -> tool calls -> tool results -> model ... until no tool calls."""
    by_name = {t.name: t for t in tools}
    records: list[ToolCallRecord] = []
    for _ in range(max_iter):
        ai: AIMessage = await gw.invoke_with_tools("worker", messages, tools)
        messages.append(ai)
        if not ai.tool_calls:
            return message_text(ai).strip(), records
        for call in ai.tool_calls:
            rec = ToolCallRecord(name=call["name"], args=call.get("args", {}))
            t0 = time.perf_counter()
            tool = by_name.get(call["name"])
            with span("tool.call", tool=call["name"]):
                try:
                    if tool is None:
                        raise ValueError(f"unknown tool {call['name']}")
                    if (tool.metadata or {}).get("side_effect") and not allow_side_effects:
                        raise PermissionError("side-effecting tool blocked by JEV policy: needs human approval")
                    out = await tool.ainvoke(call.get("args", {}))
                    rec.output = str(out)[:4000]
                except Exception as e:
                    rec.error = f"{type(e).__name__}: {e}"
            rec.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            records.append(rec)
            messages.append(ToolMessage(content=rec.output if rec.error is None else f"ERROR: {rec.error}",
                                        tool_call_id=call["id"]))
    return "", records  # ran out of iterations


class ToolsExecutor:
    SYSTEM = ("You can call tools. Use them for anything that needs live data, arithmetic or an action. "
              "Never invent tool results. When you have what you need, answer the user concisely and "
              "mention which tool results you relied on.")

    def __init__(self, gw: LLMGateway, tools: list[BaseTool], s: Settings):
        self.gw, self.tools, self.s = gw, tools, s

    async def run(self, query: str, decision: RouteDecision, strategy: dict[str, Any]) -> ExecutionResult:
        t0 = time.perf_counter()
        msgs: list[BaseMessage] = [SystemMessage(self.SYSTEM + _feedback_block(strategy)), HumanMessage(query)]
        with span("exec.tools"):
            answer, records = await run_tool_loop(
                self.gw, self.tools, msgs, self.s.tool_max_iterations,
                allow_side_effects=decision.risk < 0.6,
            )
        ctx = "\n".join(f"{r.name}({json.dumps(r.args)}) -> {r.output or r.error}" for r in records)
        return ExecutionResult(
            route=Route.TOOLS, answer=answer, context=ctx or None, tool_calls=records,
            signals={"tool_calls": len(records), "tool_errors": sum(r.error is not None for r in records),
                     "exhausted": answer == ""},
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
