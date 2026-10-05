"""
JEV post-check: Verify -> Judge -> (Accept | Retry/Re-route | Escalate).

Tiered, cheapest first. This is the LLM-as-judge cost-escalation pattern:

  tier 0  rules       free. Empty answer, tool errors, empty retrieval, citation
                      validity, PII leakage in output. If tier 0 already proves the
                      answer is unusable, we skip the judge entirely.
  tier 1  judge       cheap model, structured scores (correct/relevant/grounded/safe).
  tier 2  2nd opinion only when tier-1 lands in the gray zone. Uses the worker model.
                      Most requests never reach this tier.

Then policy decides the action, and on RETRY it also decides *how* to retry:
same route with feedback, same route with a bigger/expanded retrieval, or a
different route entirely. Every retry carries the reviewer's issues forward as
feedback so the next attempt isn't a blind re-roll.
"""
from __future__ import annotations

import logging

from app.config import Settings
from app.jev.models import (Action, CheckResult, ExecutionResult, JudgeScores, Route, RouteDecision,
                            Verdict)
from app.jev.signals import scan_pii
from app.llm.gateway import LLMGateway
from app.observability.tracing import span

log = logging.getLogger(__name__)

JUDGE_SYSTEM = """You are a strict reviewer for an AI system's answer. You do not rewrite the answer.
Score it against the request and, if provided, the CONTEXT it was supposed to rely on.
- correct: answers what was asked, nothing false
- relevant: on topic, no filler
- grounded: every specific claim traceable to CONTEXT (1.0 if no context and it's general knowledge)
- safe: no harmful instructions, no leaked personal data, no claims of actions that weren't actually done
List concrete issues. Pick suggested_fix:
  retrieve_more (answer lacks facts that docs would have), use_tools (needs live data/calc/action),
  decompose (needs multiple steps), rephrase (content is fine, answer is badly written), none."""


class PostChecker:
    def __init__(self, gw: LLMGateway, s: Settings):
        self.gw, self.s = gw, s

    # ------------------------------------------------------------------ tier 0
    def rule_checks(self, query: str, r: ExecutionResult) -> list[CheckResult]:
        checks: list[CheckResult] = []
        sig = r.signals

        has_answer = bool(r.answer.strip()) and not r.error
        checks.append(CheckResult(name="non_empty", passed=has_answer, score=1.0 if has_answer else 0.0,
                                  detail=r.error or ("" if has_answer else "empty answer")))

        leaked = [p for p in scan_pii(r.answer) if p not in scan_pii(query)]
        checks.append(CheckResult(name="output_pii", passed=not leaked, score=0.0 if leaked else 1.0,
                                  detail=f"leaked: {leaked}" if leaked else "", hard_fail=bool(leaked)))

        if r.route == Route.RAG:
            if sig.get("retrieval_empty"):
                checks.append(CheckResult(name="retrieval", passed=False, score=0.0, detail="no passages retrieved"))
            else:
                ok_cite = bool(sig.get("cited")) and not sig.get("invalid_citations")
                checks.append(CheckResult(
                    name="citations", passed=ok_cite or sig.get("abstained", False), score=1.0 if ok_cite else 0.3,
                    detail="" if ok_cite else f"cited={sig.get('cited')} invalid={sig.get('invalid_citations')}"))
                if sig.get("abstained"):
                    checks.append(CheckResult(name="abstained", passed=False, score=0.2,
                                              detail="model said the KB doesn't cover it"))

        if r.route in (Route.TOOLS, Route.AGENT):
            blocked = [t for t in r.tool_calls if t.error and "PermissionError" in t.error]
            # ValueError = the tool worked and said "no" (order not found, bad input). That's a
            # valid answer for the user, not a system failure, so it doesn't trigger a re-route;
            # the judge decides whether the answer explained it properly.
            input_errors = [t for t in r.tool_calls if t.error and t.error.startswith("ValueError")]
            errors = [t for t in r.tool_calls if t.error and t not in blocked and t not in input_errors]
            if blocked:
                checks.append(CheckResult(name="action_approval", passed=False, score=0.0, hard_fail=True,
                                          detail=f"needs approval: {[t.name for t in blocked]}"))
            checks.append(CheckResult(name="tool_health", passed=not errors, score=0.0 if errors else 1.0,
                                      detail="; ".join(f"{t.name}: {t.error}" for t in errors)))
            if r.route == Route.TOOLS and sig.get("exhausted"):
                checks.append(CheckResult(name="tool_loop", passed=False, score=0.0, detail="hit iteration limit"))
        return checks

    # ------------------------------------------------------------------ tier 1/2
    async def judge(self, query: str, r: ExecutionResult, role: str) -> JudgeScores:
        ctx = (r.context or "")[:6000]
        user = (f"REQUEST:\n{query}\n\n" + (f"CONTEXT:\n{ctx}\n\n" if ctx else "CONTEXT: (none)\n\n") +
                f"ANSWER:\n{r.answer}")
        return await self.gw.structured(role, JUDGE_SYSTEM, user, JudgeScores)  # type: ignore[arg-type]

    @staticmethod
    def aggregate(j: JudgeScores, has_context: bool) -> float:
        if has_context:
            return 0.4 * j.correct + 0.2 * j.relevant + 0.3 * j.grounded + 0.1 * j.confidence
        return 0.55 * j.correct + 0.3 * j.relevant + 0.15 * j.confidence

    # ------------------------------------------------------------------ reroute policy
    def plan_retry(self, decision: RouteDecision, r: ExecutionResult, checks: list[CheckResult],
                   judge: JudgeScores | None, history: list[dict]) -> tuple[Route | None, dict]:
        tried = {(h["route"], h.get("strategy_key", "")) for h in history}
        failed = {c.name for c in checks if not c.passed}
        fix = judge.suggested_fix if judge else "none"
        feedback = (judge.issues if judge else []) + [f"{c.name}: {c.detail}" for c in checks if not c.passed and c.detail]

        candidates: list[tuple[Route, dict]] = []
        route = r.route
        if route == Route.RAG and failed & {"retrieval", "abstained", "citations"}:
            candidates += [(Route.RAG, {"expand": True, "k": self.s.retrieval_k * 2}), (Route.AGENT, {})]
        if route == Route.LLM_ONLY and (fix == "retrieve_more" or (judge and judge.grounded < 0.6)):
            candidates.append((Route.RAG, {}))
        if fix == "retrieve_more" and route != Route.LLM_ONLY:
            candidates.append((Route.RAG, {"expand": True, "k": self.s.retrieval_k * 2}))
        if fix == "use_tools" or (route == Route.LLM_ONLY and decision.signals.action_cues):
            candidates.append((Route.TOOLS, {}))
        if route == Route.TOOLS and failed & {"tool_health", "tool_loop"}:
            candidates.append((Route.AGENT, {}))
        if fix == "decompose":
            candidates.append((Route.AGENT, {}))
        # generic fallback: same route with reviewer feedback, then the most capable path
        candidates += [(route, {"retry_same": True}), (Route.AGENT, {}), (Route.RAG, {"expand": True})]

        for nr, strat in candidates:
            key = ",".join(f"{k}={v}" for k, v in sorted(strat.items()))
            if (nr.value, key) not in tried:
                return nr, {**strat, "feedback": feedback[:6], "strategy_key": key}
        return None, {}

    # ------------------------------------------------------------------ main
    async def check(self, query: str, decision: RouteDecision, r: ExecutionResult,
                    attempt: int, history: list[dict], max_attempts: int | None = None) -> Verdict:
        max_attempts = max_attempts or self.s.max_attempts
        with span("jev.post_check", route=r.route.value, attempt=attempt) as attrs:
            checks = self.rule_checks(query, r)
            hard = [c for c in checks if c.hard_fail and not c.passed]
            unusable = not checks[0].passed  # non_empty failed

            judge, tier = None, 0
            if not hard and not unusable:
                try:
                    judge, tier = await self.judge(query, r, "judge"), 1
                    has_ctx = bool(r.context)
                    lo, hi = self.s.judge_gray_zone
                    s1 = self.aggregate(judge, has_ctx)
                    if self.s.enable_judge_escalation and lo <= s1 < hi:
                        judge, tier = await self.judge(query, r, "worker"), 2
                except Exception as e:
                    log.warning("judge failed: %s", e)
                    checks.append(CheckResult(name="judge", passed=False, score=0.5, detail=f"judge error: {e}"))

            if judge:
                score = self.aggregate(judge, bool(r.context))
                checks.append(CheckResult(name="judge_safe", passed=judge.safe, score=1.0 if judge.safe else 0.0,
                                          hard_fail=not judge.safe, detail="; ".join(judge.issues[:3])))
            else:
                score = sum(c.score for c in checks) / len(checks) * 0.6  # rules alone can't earn an accept

            soft_fails = [c for c in checks if not c.passed and not c.hard_fail]
            if soft_fails:
                score = min(score, 0.6)
            score = round(score, 3)
            hard = [c for c in checks if c.hard_fail and not c.passed]

            # ---------------- decision ----------------
            next_route, strategy = None, {}
            if hard:
                action, reason = Action.ESCALATE, "hard fail: " + ", ".join(c.name for c in hard)
            elif score >= self.s.accept_threshold and not soft_fails:
                action, reason = Action.ACCEPT, f"score {score} >= {self.s.accept_threshold}"
            elif attempt >= max_attempts:
                action, reason = Action.ESCALATE, f"still failing after {attempt} attempts (score {score})"
            else:
                next_route, strategy = self.plan_retry(decision, r, checks, judge, history)
                if next_route is None:
                    action, reason = Action.ESCALATE, "no untried recovery path left"
                else:
                    action = Action.RETRY
                    reason = f"score {score}; retry via {next_route.value} {strategy.get('strategy_key') or ''}".strip()

            attrs.update(action=action.value, score=score, judge_tier=tier,
                         failed=",".join(c.name for c in checks if not c.passed))
            return Verdict(action=action, score=score, checks=checks, judge=judge, judge_tier=tier,
                           next_route=next_route, next_strategy=strategy, reason=reason)
