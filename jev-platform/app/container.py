"""Builds every service once at startup. Tests build it with a fake gateway."""
from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.executors.agent_workflow import AgentWorkflowExecutor
from app.executors.basic import LLMOnlyExecutor, RAGExecutor, ToolsExecutor
from app.graph.jev_graph import build_jev_graph
from app.jev.escalation import EscalationQueue
from app.jev.models import Route
from app.jev.post_checker import PostChecker
from app.jev.pre_router import PreRouter
from app.llm.gateway import LLMGateway
from app.observability.tracing import TraceStore, init_trace_store
from app.rag.embeddings import build_embedder
from app.rag.retriever import HybridRetriever, IngestService
from app.rag.stores import VectorStore, build_store
from app.tools.registry import build_tools


@dataclass
class Container:
    settings: Settings
    gateway: LLMGateway
    store: VectorStore
    retriever: HybridRetriever
    ingest: IngestService
    escalations: EscalationQueue
    traces: TraceStore
    graph: object
    tool_names: list[str]


def build_container(settings: Settings, gateway: LLMGateway | None = None) -> Container:
    gw = gateway or LLMGateway(settings)
    embedder = build_embedder(settings)
    store = build_store(settings, embedder.dim)
    retriever = HybridRetriever(store, embedder, gw, settings)
    tools = build_tools(retriever)
    executors = {
        Route.LLM_ONLY: LLMOnlyExecutor(gw),
        Route.RAG: RAGExecutor(gw, retriever, settings),
        Route.TOOLS: ToolsExecutor(gw, tools, settings),
        Route.AGENT: AgentWorkflowExecutor(gw, tools, settings),
    }
    escalations = EscalationQueue()
    graph = build_jev_graph(PreRouter(gw, settings), PostChecker(gw, settings), executors, escalations)
    return Container(settings=settings, gateway=gw, store=store, retriever=retriever,
                     ingest=IngestService(store, embedder, settings), escalations=escalations,
                     traces=init_trace_store(settings.trace_store_size), graph=graph,
                     tool_names=[t.name for t in tools])
