"""
JEV pre-routing: Classify -> Score -> Prioritize -> Route.

Two inputs get blended:
  * deterministic signals (signals.py)          weight = 1 - router_llm_weight
  * an LLM classifier with structured output     weight = router_llm_weight

Then policy overrides run on top. Overrides always win over scores, because a
score is a guess and a policy is a rule someone signed off on.

If the classifier call fails or times out we don't fail the request; we route on
heuristics alone and record classifier_error so it shows up in the trace.
"""
from __future__ import annotations

import asyncio
import logging

from app.config import Settings
from app.jev.models import Priority, Route, RouteClassification, RouteDecision, Signals
from app.jev.signals import extract_signals, heuristic_scores
from app.llm.gateway import LLMGateway
from app.observability.tracing import span

log = logging.getLogger(__name__)

ROUTER_SYSTEM = """You are the routing classifier for an AI platform. You never answer the user.
You score the request so the platform can pick an execution path:

- llm_only : general reasoning or writing, no private data, no actions
- rag      : needs the company knowledge base (policies, docs, product info, procedures)
- tools    : needs a single tool/API: look up an order, calculate, create a ticket, get the time
- agent    : needs several dependent steps that mix lookups, knowledge and actions

Score every field between 0 and 1. Be calibrated: 0.5 means unsure.
risk is about the cost of a wrong answer or wrong action, not about tone."""

ROUTE_KEYS = [Route.LLM_ONLY, Route.RAG, Route.TOOLS, Route.AGENT]


class PreRouter:
    def __init__(self, gateway: LLMGateway, settings: Settings):
        self.gw = gateway
        self.s = settings

    async def classify(self, query: str) -> RouteClassification:
        return await asyncio.wait_for(
            self.gw.structured("router", ROUTER_SYSTEM, f"Request:\n{query}", RouteClassification),
            timeout=self.s.llm_timeout_s,
        )

    @staticmethod
    def llm_scores(c: RouteClassification) -> dict[str, float]:
        k, a, x = c.needs_knowledge, c.needs_action, c.complexity
        return {
            "llm_only": (1 - k) * (1 - a) * (1 - x),
            "rag": k * (1 - a) * (1 - 0.6 * x),
            "tools": a * (1 - 0.6 * x) * (1 - 0.4 * k),
            "agent": x * max(k, a),
        }

    def _priority(self, sig: Signals, risk: float) -> Priority:
        if sig.urgency or risk >= 0.7:
            return Priority.HIGH
        if risk <= 0.2 and not sig.action_cues:
            return Priority.LOW
        return Priority.NORMAL

    async def decide(self, query: str, force_route: Route | None = None) -> RouteDecision:
        with span("jev.pre_route") as attrs:
            sig = extract_signals(query)
            h = heuristic_scores(sig)

            classifier, err = None, None
            if sig.injection_suspected:
                err = "skipped: injection pattern, request never sent to a model"
            else:
                try:
                    classifier = await self.classify(query)
                except Exception as e:  # network, schema, timeout: degrade, don't die
                    err = f"{type(e).__name__}: {e}"
                    log.warning("router classifier failed, using heuristics only: %s", err)

            if classifier:
                l = self.llm_scores(classifier)
                w = self.s.router_llm_weight
                scores = {r: round(w * l[r] + (1 - w) * h[r], 4) for r in l}
                risk = max(classifier.risk, 0.6 if sig.risky_action else 0.0)
                intent = classifier.intent
            else:
                scores = {r: round(v, 4) for r, v in h.items()}
                risk = 0.6 if sig.risky_action else 0.2
                intent = "unknown"

            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            best, best_s = ranked[0]
            second_s = ranked[1][1] if len(ranked) > 1 else 0.0
            # margin-based confidence: a 0.40 vs 0.38 split is a coin flip even if 0.40 looks decent
            confidence = round(min(1.0, (best_s - second_s) / max(best_s, 1e-6)), 3)
            route = Route(best)
            reason = f"highest blended score ({best}={best_s:.2f}, margin {confidence:.2f})"

            # Low confidence between a "plain" route and a richer one: take the richer
            # one. Retrieval you didn't need costs a few hundred ms; skipping retrieval you
            # did need costs a wrong answer.
            if confidence < 0.15 and route == Route.LLM_ONLY and ranked[1][0] == "rag":
                route, reason = Route.RAG, "low-confidence llm_only vs rag tie, preferring grounded path"

            # ---------------- policy overrides (always win) ----------------
            if sig.injection_suspected:
                route, reason = Route.ESCALATE, "prompt-injection pattern in request"
            elif risk >= self.s.risk_escalation_threshold:
                route, reason = Route.ESCALATE, f"risk {risk:.2f} above threshold {self.s.risk_escalation_threshold}"
            elif force_route:
                route, reason = force_route, "forced by caller"

            decision = RouteDecision(
                route=route,
                intent=intent,
                priority=self._priority(sig, risk),
                confidence=confidence,
                scores=scores,
                risk=round(risk, 3),
                signals=sig,
                classifier=classifier,
                classifier_error=err,
                reason=reason,
            )
            attrs.update(route=route.value, intent=intent, confidence=confidence, risk=decision.risk,
                         priority=decision.priority.value, reason=reason)
            return decision
