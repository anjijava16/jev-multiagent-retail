"""Runtime settings. Everything is read from the environment (or a .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass

try:  # optional dependency
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def use_mock_jev() -> bool:
    if os.getenv("JEV_MOCK") == "1":
        return True
    return not os.getenv("TYPESAFE_API_KEY", "").strip()


def use_mock_llm() -> bool:
    flag = os.getenv("LLM_MOCK", "auto").lower()
    if flag in ("1", "true", "yes"):
        return True
    if flag in ("0", "false", "no"):
        return False
    keys = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY")
    return not any(os.getenv(k, "").strip() for k in keys)


MODELS = {
    "product_agent": os.getenv("PRODUCT_AGENT_MODEL", "openai/gpt-4o-mini"),
    "returns_agent": os.getenv("RETURNS_AGENT_MODEL", "anthropic/claude-sonnet-4-5"),
    "recommend_agent": os.getenv("RECOMMEND_AGENT_MODEL", "gemini/gemini-2.5-flash"),
    "complaint_agent": os.getenv("COMPLAINT_AGENT_MODEL", "anthropic/claude-sonnet-4-5"),
    "synth": os.getenv("SYNTH_MODEL", "openai/gpt-4o-mini"),
}


@dataclass(frozen=True)
class Thresholds:
    """Every number the system acts on lives here, not inside a prompt.

    Change a threshold, rerun the eval set, compare. That's the whole loop.
    """

    # guardrails (Noul probabilities)
    injection: float = 0.7
    off_topic: float = 0.8
    # triage
    intent_floor: float = 0.5          # Choice confidence below this -> human
    wants_human: float = 0.7           # Noul
    multi_request: float = 0.6         # Noul
    secondary_p: float = 0.15          # min probability for a 2nd intent to get its own agent
    escalate_complexity: float = 1.5   # Score on a 0-2 scale ...
    escalate_frustration: float = 1.5  # ... both above this -> straight to a human
    # returns
    high_value_refund: float = 200.0
    refund_auto_conf: float = 0.8
    # complaints
    safety: float = 0.5
    legal: float = 0.6
    # output verification
    grounded: float = 0.6


TH = Thresholds()
