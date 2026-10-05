"""Pure routing logic. No I/O, no network: give it Jev answers, get a plan back.

This is the heart of the "Jev orchestrator" idea. Jev supplies calibrated
answers; this function turns them into a decision with thresholds you own.
Because it's pure, it's trivially unit-testable (see tests/test_router.py).
"""
from __future__ import annotations

from .config import TH, Thresholds
from .jev_client import Answer

INTENT_TO_AGENT = {
    "order_status": "order_agent",
    "product_question": "product_agent",
    "recommendation": "recommend_agent",
    "return_refund": "returns_agent",
    "complaint": "complaint_agent",
}
HUMAN = "human_handoff"


def plan_route(triage: dict[str, Answer], th: Thresholds = TH) -> tuple[list[str], str]:
    intent = triage["intent"]
    wants_human = triage["wants_human"].noul or 0.0
    complexity = triage["complexity"]
    frustration = triage["frustration"]

    # 1. the customer asked for a person: respect that, no clever routing
    if wants_human >= th.wants_human:
        return [HUMAN], f"customer asked for a person (p={wants_human:.2f})"

    # 2. Jev isn't sure what they want: don't guess.
    #    Exception: a message that really asks for two things splits the intent
    #    probability between them. That's not confusion, so check the top two
    #    together before calling it uncertain.
    multi = triage["multi_request"].noul or 0.0
    top2 = sum(p for _, p in intent.top(2))
    split_but_clear = multi >= th.multi_request and top2 >= 0.7
    if (intent.confidence or 0) < th.intent_floor and not split_but_clear:
        return [HUMAN], f"intent unclear ({intent.choice}, confidence {intent.confidence:.2f})"

    # 3. hard and heated: a human will do better than any agent
    if (complexity.score or 0) >= th.escalate_complexity and (frustration.score or 0) >= th.escalate_frustration:
        return [HUMAN], f"complex ({complexity.score:.1f}) and upset ({frustration.score:.1f})"

    if intent.choice not in INTENT_TO_AGENT:
        return [HUMAN], f"no agent for intent '{intent.choice}'"

    plan = [INTENT_TO_AGENT[intent.choice]]
    reason = f"intent={intent.choice} (conf {intent.confidence:.2f})"

    # 4. multi-request: use the probability distribution, not just the top answer
    if multi >= th.multi_request:
        for option, p in intent.top(len(intent.probabilities)):
            agent = INTENT_TO_AGENT.get(option)
            if option != intent.choice and agent and p >= th.secondary_p and agent not in plan:
                plan.append(agent)
                reason += f" + {option} (p={p:.2f})"
    return plan, reason
