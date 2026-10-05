from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.api.schemas import (AskRequest, AskResponse, AttemptSummary, IngestRequest, IngestResponse,
                             ResolveRequest, RetrieveRequest)
from app.container import Container
from app.observability.tracing import current_trace_id

router = APIRouter()


def _c(request: Request) -> Container:
    return request.app.state.container


# =========================================================================== ask
@router.post("/v1/ask", response_model=AskResponse, tags=["jev"],
             summary="Send a request through JEV (pre-route -> execute -> post-check)")
async def ask(body: AskRequest, request: Request) -> AskResponse:
    c = _c(request)
    trace_id = uuid.uuid4().hex
    token = current_trace_id.set(trace_id)
    c.traces.start(trace_id, body.model_dump(mode="json"))
    t0 = time.perf_counter()
    try:
        st = await c.graph.ainvoke(
            {"trace_id": trace_id, "query": body.query, "user_id": body.user_id, "filters": body.filters,
             "force_route": body.force_route, "max_attempts": body.max_attempts, "history": []},
            config={"recursion_limit": 40, "run_name": "jev", "metadata": {"trace_id": trace_id}},
        )
    finally:
        current_trace_id.reset(token)

    d, v, r = st["decision"], st.get("verdict"), st.get("result")
    latency = round((time.perf_counter() - t0) * 1000, 1)
    resp = AskResponse(
        trace_id=trace_id,
        status=st["status"],
        answer=st["final_answer"],
        route_taken=r.route.value if r else None,
        initial_route=d.route.value,
        intent=d.intent,
        priority=d.priority.value,
        attempts=[AttemptSummary(**h) for h in st.get("history", [])],
        final_score=v.score if v else None,
        citations=r.citations if (r and st["status"] == "answered") else [],
        tool_calls=r.tool_calls if r else [],
        plan=r.plan if r else [],
        escalation_id=st.get("escalation_id"),
        latency_ms=latency,
        debug={"pre_route": d.model_dump(mode="json"), "verdict": v.model_dump(mode="json") if v else None}
        if body.debug else None,
    )
    c.traces.finish(trace_id, {"status": resp.status, "route": resp.route_taken, "score": resp.final_score})
    return resp


@router.post("/v1/route", tags=["jev"], summary="Dry-run: show the pre-route decision without executing")
async def route_only(body: AskRequest, request: Request):
    from app.jev.pre_router import PreRouter
    c = _c(request)
    d = await PreRouter(c.gateway, c.settings).decide(body.query, body.force_route)
    return d.model_dump(mode="json")


# =========================================================================== knowledge
@router.post("/v1/ingest", response_model=IngestResponse, tags=["knowledge"])
async def ingest(body: IngestRequest, request: Request) -> IngestResponse:
    c = _c(request)
    total = 0
    for d in body.documents:
        total += await c.ingest.ingest_text(d.text, d.source, d.title, d.metadata)
    return IngestResponse(documents=len(body.documents), chunks=total)


@router.post("/v1/ingest/file", response_model=IngestResponse, tags=["knowledge"],
             summary="Upload .md / .txt files")
async def ingest_file(request: Request, files: list[UploadFile] = File(...),
                      department: str | None = Form(default=None)) -> IngestResponse:
    c = _c(request)
    total = 0
    for f in files:
        if not (f.filename or "").lower().endswith((".md", ".txt")):
            raise HTTPException(400, f"{f.filename}: only .md and .txt are supported here")
        text = (await f.read()).decode("utf-8", errors="replace")
        meta = {"department": department} if department else {}
        total += await c.ingest.ingest_text(text, f.filename, None, meta)
    return IngestResponse(documents=len(files), chunks=total)


@router.post("/v1/retrieve", tags=["knowledge"], summary="Debug retrieval without generation")
async def retrieve(body: RetrieveRequest, request: Request):
    hits = await _c(request).retriever.retrieve(body.query, body.k, body.filters, body.expand)
    return [{k: v for k, v in h.items() if k != "embedding"} for h in hits]


# =========================================================================== ops
@router.get("/v1/traces/{trace_id}", tags=["ops"])
async def get_trace(trace_id: str, request: Request):
    t = _c(request).traces.get(trace_id)
    if not t:
        raise HTTPException(404, "trace not found (traces are in-memory and capped)")
    return t


@router.get("/v1/escalations", tags=["ops"])
async def list_escalations(request: Request, status: str | None = None):
    return [e.model_dump() for e in _c(request).escalations.list(status)]


@router.post("/v1/escalations/{esc_id}/resolve", tags=["ops"])
async def resolve_escalation(esc_id: str, body: ResolveRequest, request: Request):
    e = _c(request).escalations.resolve(esc_id, body.resolution)
    if not e:
        raise HTTPException(404, "escalation not found")
    return e.model_dump()


@router.get("/health", tags=["ops"])
async def health(request: Request):
    c = _c(request)
    return {
        "status": "ok",
        "llm_provider": c.settings.llm_provider,
        "models": {r: c.gateway.model_name(r) for r in ("router", "worker", "judge")},
        "vector_backend": c.settings.vector_backend,
        "vector_store_up": await c.store.ping(),
        "indexed_chunks": await c.store.count(),
        "tools": c.tool_names,
    }
