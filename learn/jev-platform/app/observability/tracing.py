"""
Lightweight tracing.

Every request gets a trace_id. Each JEV stage records a span (name, start, duration,
attributes). Spans go to:
  1. an in-memory ring buffer you can read back at GET /v1/traces/{trace_id}
  2. OpenTelemetry, if `opentelemetry-api` is installed and a provider is configured
     (Monocle, Phoenix, Jaeger, anything OTLP). No hard dependency.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

try:  # optional
    from opentelemetry import trace as _otel

    _tracer = _otel.get_tracer("jev")
except Exception:  # pragma: no cover
    _tracer = None

current_trace_id: ContextVar[str | None] = ContextVar("current_trace_id", default=None)


class TraceStore:
    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self._traces: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def start(self, trace_id: str, request: dict[str, Any]) -> None:
        self._traces[trace_id] = {"trace_id": trace_id, "request": request, "spans": [], "started": time.time()}
        while len(self._traces) > self.max_size:
            self._traces.popitem(last=False)

    def add_span(self, trace_id: str, span: dict[str, Any]) -> None:
        if trace_id in self._traces:
            self._traces[trace_id]["spans"].append(span)

    def finish(self, trace_id: str, outcome: dict[str, Any]) -> None:
        if trace_id in self._traces:
            t = self._traces[trace_id]
            t["outcome"] = outcome
            t["duration_ms"] = round((time.time() - t["started"]) * 1000, 1)

    def get(self, trace_id: str) -> dict[str, Any] | None:
        return self._traces.get(trace_id)


_store: TraceStore | None = None


def init_trace_store(size: int) -> TraceStore:
    global _store
    _store = TraceStore(size)
    return _store


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
    """Usage:  with span("jev.pre_route", query=q) as s: ...; s["route"] = "rag" """
    record: dict[str, Any] = {"name": name, "attrs": dict(attrs)}
    t0 = time.perf_counter()
    otel_cm = _tracer.start_as_current_span(name) if _tracer else None
    otel_span = otel_cm.__enter__() if otel_cm else None
    try:
        yield record["attrs"]
        record["status"] = "ok"
    except Exception as e:
        record["status"] = "error"
        record["error"] = repr(e)
        raise
    finally:
        record["duration_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if otel_span is not None:
            for k, v in record["attrs"].items():
                if isinstance(v, (str, int, float, bool)):
                    otel_span.set_attribute(f"jev.{k}", v)
            otel_cm.__exit__(None, None, None)
        tid = current_trace_id.get()
        if _store and tid:
            _store.add_span(tid, record)
