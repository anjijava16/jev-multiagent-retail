"""
FastAPI entrypoint.   uvicorn app.main:app --reload   ->   http://localhost:8000/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

from app.api.routes import router
from app.config import get_settings
from app.container import build_container


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    logging.basicConfig(level=s.log_level, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    # tests may pre-install a container with a fake gateway
    if not getattr(app.state, "container", None):
        app.state.container = build_container(s)
    try:
        await app.state.container.store.ensure_index()
    except Exception as e:  # don't block startup; /health will show the store is down
        logging.getLogger("jev").error("vector store not ready: %s", e)
    yield


app = FastAPI(
    title="JEV Platform",
    version="1.0.0",
    description=(
        "JEV = the decision + evaluation layer around an AI system.\n\n"
        "**Before execution** it classifies, scores, prioritizes and routes a request to "
        "LLM-only, RAG (OpenSearch), Tools, or a LangGraph agent workflow.\n\n"
        "**After execution** it verifies and judges the result, then accepts it, "
        "retries / re-routes it, or escalates it to a human."
    ),
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/v1/graph", response_class=PlainTextResponse, tags=["ops"],
         summary="The live JEV LangGraph as Mermaid (paste into mermaid.live)")
async def graph_mermaid(request: Request) -> str:
    return request.app.state.container.graph.get_graph().draw_mermaid()
