"""Human-in-the-loop queue. In-memory here; in production this is a table + a reviewer UI or a ticketing system."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class Escalation(BaseModel):
    id: str
    trace_id: str
    query: str
    reason: str
    priority: str
    draft_answer: str | None = None      # what the system would have said, for the reviewer
    route_history: list[dict[str, Any]] = Field(default_factory=list)
    status: str = "open"                 # open | resolved
    resolution: str | None = None
    created: str
    resolved: str | None = None


class EscalationQueue:
    def __init__(self):
        self._items: dict[str, Escalation] = {}

    def open(self, **kw) -> Escalation:
        e = Escalation(id=f"ESC-{uuid.uuid4().hex[:8].upper()}",
                       created=datetime.now(timezone.utc).isoformat(), **kw)
        self._items[e.id] = e
        return e

    def list(self, status: str | None = None) -> list[Escalation]:
        items = list(self._items.values())
        return [e for e in items if status is None or e.status == status]

    def get(self, esc_id: str) -> Escalation | None:
        return self._items.get(esc_id)

    def resolve(self, esc_id: str, resolution: str) -> Escalation | None:
        e = self._items.get(esc_id)
        if e:
            e.status, e.resolution = "resolved", resolution
            e.resolved = datetime.now(timezone.utc).isoformat()
        return e
