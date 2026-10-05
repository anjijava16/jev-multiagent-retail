"""
Tier-0 signals: regex and keyword checks that cost nothing.

These run before any LLM call (pre-route) and after execution (post-check).
They are deliberately conservative. They don't decide the route on their own;
they nudge the blended score and can trigger hard policy overrides
(prompt injection, PII leakage, destructive actions).
"""
from __future__ import annotations

import re

from app.jev.models import Signals

PII_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "card_number": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
}

INJECTION_PATTERNS = [
    r"ignore (all|any|the)? ?(previous|prior|above) (instructions|rules)",
    r"disregard (your|the) (system|previous) prompt",
    r"you are now (dan|in developer mode)",
    r"reveal (your|the) (system prompt|instructions)",
    r"print (your|the) system prompt",
]
_INJECTION = re.compile("|".join(INJECTION_PATTERNS), re.I)

ACTION_WORDS = [
    "create", "open a ticket", "file a", "update", "change", "cancel", "refund",
    "send", "book", "schedule", "delete", "remove", "lookup", "look up", "check status",
    "track", "calculate", "compute", "convert", "what time",
]
RISKY_ACTION_WORDS = ["delete", "cancel", "refund", "transfer", "wire", "close my account", "remove"]

KNOWLEDGE_WORDS = [
    "policy", "policies", "according to", "handbook", "documentation", "docs",
    "our ", "internal", "sla", "guideline", "procedure", "terms", "warranty",
    "return window", "what does the", "faq",
]

MULTI_STEP_WORDS = [
    " and then ", "after that", "step by step", "compare", "for each", "first ",
    " then ", "both ", "and also", "plan ",
]

URGENCY_WORDS = ["urgent", "asap", "immediately", "right now", "emergency"]

_MATH = re.compile(r"\d+\s*[\+\-\*/x%^]\s*\d+|\b(percent|percentage|sum of|average of|multiply)\b", re.I)


def _hits(text: str, words: list[str]) -> list[str]:
    t = f" {text.lower()} "
    return [w.strip() for w in words if w in t]


def scan_pii(text: str) -> list[str]:
    found = []
    for name, pat in PII_PATTERNS.items():
        if pat.search(text):
            found.append(name)
    return found


def extract_signals(query: str) -> Signals:
    return Signals(
        pii_found=scan_pii(query),
        injection_suspected=bool(_INJECTION.search(query)),
        action_cues=_hits(query, ACTION_WORDS),
        knowledge_cues=_hits(query, KNOWLEDGE_WORDS),
        multi_step_cues=_hits(query, MULTI_STEP_WORDS),
        has_math=bool(_MATH.search(query)),
        urgency=bool(_hits(query, URGENCY_WORDS)),
        risky_action=bool(_hits(query, RISKY_ACTION_WORDS)),
    )


def heuristic_scores(sig: Signals) -> dict[str, float]:
    """Rough per-route prior from signals alone. Used when the classifier is down, and blended otherwise."""
    knowledge = min(1.0, 0.35 * len(sig.knowledge_cues))
    action = min(1.0, 0.4 * len(sig.action_cues) + (0.5 if sig.has_math else 0.0))
    multi = min(1.0, 0.35 * len(sig.multi_step_cues))
    return {
        "llm_only": max(0.0, 0.6 - max(knowledge, action, multi)),
        "rag": knowledge * (1 - 0.5 * action),
        "tools": action * (1 - 0.5 * multi),
        "agent": multi * max(knowledge, action, 0.3),
    }


CITATION_RE = re.compile(r"\[(\d{1,2})\]")


def cited_ids(answer: str) -> set[int]:
    return {int(m) for m in CITATION_RE.findall(answer)}
