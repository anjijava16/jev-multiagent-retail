"""Thin wrapper around the TypeSafe SDK.

Why wrap it at all?
  1. One place to swap the real API for the offline mock.
  2. Answers come back as a small plain dataclass, so they can live in LangGraph
     state (which wants plain, serialisable values) and be unit-tested easily.
  3. A call counter and latency log, so you can see how cheap the routing is.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from . import config


@dataclass
class Answer:
    type: str                       # "choice" | "score" | "noul"
    choice: str | None = None       # Choice
    score: float | None = None      # Score (expected level, 0..n-1)
    noul: float | None = None       # Noul (probability the statement is true)
    confidence: float | None = None  # Choice and Score only
    probabilities: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Answer":
        return cls(**d)

    def top(self, n: int = 3) -> list[tuple[str, float]]:
        return sorted(self.probabilities.items(), key=lambda kv: kv[1], reverse=True)[:n]


def _get(obj: Any, key: str):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _normalize(raw: Any) -> Answer:
    probs = _get(raw, "probabilities") or {}
    probs = {str(k): float(v) for k, v in dict(probs).items()}
    kind = _get(raw, "type")
    if kind is None:  # defensive: infer from fields
        kind = "choice" if _get(raw, "choice") is not None else "score" if _get(raw, "score") is not None else "noul"
    num = lambda v: None if v is None else float(v)  # noqa: E731
    return Answer(
        type=str(kind),
        choice=_get(raw, "choice"),
        score=num(_get(raw, "score")),
        noul=num(_get(raw, "noul")),
        confidence=num(_get(raw, "confidence")),
        probabilities=probs,
    )


class Jev:
    def __init__(self) -> None:
        if config.use_mock_jev():
            from .mock_jev import MockJev

            self._client = MockJev()
            self.mode = "mock"
        else:
            from typesafe_sdk import TypeSafeClient  # reads TYPESAFE_API_KEY

            self._client = TypeSafeClient()
            self.mode = "live"
        self.calls = 0
        self.log: list[dict] = []

    def ask(self, state: Any, questions: dict[str, dict], *, tag: str = "") -> dict[str, Answer]:
        """Send one state + N questions, get N typed answers back."""
        t0 = time.perf_counter()
        resp = self._client.system_one(state=state, questions=questions)
        ms = (time.perf_counter() - t0) * 1000
        self.calls += 1
        self.log.append({"tag": tag, "questions": len(questions), "ms": round(ms, 1)})
        answers = _get(resp, "answers") or {}
        return {name: _normalize(a) for name, a in dict(answers).items()}


_JEV: Jev | None = None


def get_jev() -> Jev:
    global _JEV
    if _JEV is None:
        _JEV = Jev()
    return _JEV


def reset_jev() -> None:
    global _JEV
    _JEV = None
