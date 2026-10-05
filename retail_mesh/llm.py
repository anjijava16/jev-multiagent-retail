"""Text generation for the sub-agents.

Jev never writes text. When an agent needs a customer-facing reply, it calls an
ordinary LLM through LiteLLM, so each agent can sit on a different provider
(OpenAI, Anthropic, Gemini, Bedrock, a local Ollama model, ...).

Every call passes a `fallback`: a plain template built from the same facts.
It's used in mock mode, and also if the provider errors out. A customer should
never get a stack trace because a model was down.
"""
from __future__ import annotations

import logging

from . import config

log = logging.getLogger(__name__)

GROUNDING_RULES = (
    "You are a customer support writer for ShopSense, an online retailer.\n"
    "Write a short, friendly reply (max 120 words).\n"
    "Only use facts from the FACTS block. Do not invent order numbers, prices, dates, "
    "discounts or policies. If something is not in FACTS, say you will check, don't guess.\n"
    "Plain text, no markdown headings."
)


def complete(model: str, task: str, facts: dict, fallback: str) -> tuple[str, str]:
    """Return (text, source) where source is the model name or 'template'."""
    if config.use_mock_llm():
        return fallback, "template"
    try:
        import litellm

        resp = litellm.completion(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": GROUNDING_RULES},
                {"role": "user", "content": f"TASK:\n{task}\n\nFACTS:\n{facts}"},
            ],
        )
        return resp.choices[0].message.content.strip(), model
    except Exception as exc:  # provider down, bad key, rate limit...
        log.warning("LLM call to %s failed (%s); using template", model, exc)
        return fallback, "template"
