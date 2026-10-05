"""A keyword-based stand-in for Jev so the whole project runs with no API key.

It is NOT a model. It exists so you can run the graph, read the trace and learn
the shapes. It returns exactly the same answer shapes as the real API
(choice/score/noul + probabilities + confidence, using the formulas from
https://docs.typesafe.ai/confidence), so nothing else in the code changes when
you set TYPESAFE_API_KEY.
"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any

# ----------------------------------------------------------------- vocabulary
HINTS: dict[str, list[str]] = {
    # intents
    "order_status": ["where is", "track", "tracking", "shipped", "delivery", "hasn't arrived",
                     "check on", "status of", "eta", "hasn't come", "supposed to come"],
    "product_question": ["do you have", "you sell", "in stock", "good for", "does it", "specs",
                         "you carry"],
    "recommendation": ["gift", "recommend", "suggest", "ideas", "budget", "for my"],
    "return_refund": ["return", "refund", "exchange", "send it back", "money back", "doesn't fit",
                      "stopped", "not working"],
    "complaint": ["unacceptable", "disappointed", "terrible", "worst", "complain", "snapped",
                  "cut my", "torn", "missing", "problems", "ridiculous"],
    "other": [],
    # return reasons
    "defective": ["stopped", "broken", "not working", "doesn't work", "won't charge", "died", "fault",
                  "crackle", "since day one"],
    "damaged_in_shipping": ["arrived damaged", "crushed", "arrived broken", "dented"],
    "wrong_item": ["wrong item", "not what i ordered", "sent me the wrong"],
    "size_fit": ["fit", "size", "too small", "too big", "tight", "loose"],
    "changed_mind": ["changed my mind", "don't need", "no longer want"],
    # resolutions
    "refund": ["refund", "money back"],
    "exchange": ["exchange", "replace", "replacement", "swap"],
    "store_credit": ["store credit", "credit"],
}

ANGER = ["unacceptable", "ridiculous", "terrible", "worst", "furious", "angry", "disappointed", "!!"]
SAFETY = ["fire", "burn", "smoke", "shock", "injur", "cut my", "bleed", "allergic", "hurt"]
LEGAL = ["lawyer", "attorney", "sue", "chargeback", "legal", "bbb", "regulator"]
INJECTION = ["ignore all previous", "ignore previous", "ignore your instructions", "system prompt",
             "you are now", "developer mode", "100% discount", "free discount code"]
ABUSE = ["idiot", "stupid", "useless bot", "shut up"]
RETAIL = ["order", "buy", "bought", "return", "refund", "product", "sell", "price", "ship", "deliver",
          "gift", "store", "item", "stock", "discount", "earbud", "jacket", "tent", "kettle", "coffee",
          "headphone", "shoe", "charger", "lamp", "mat", "sweater", "track", "exchange", "recommend"]

_WORD = re.compile(r"[a-z0-9][a-z0-9\-]+")
_ORDER = re.compile(r"\bSS-\d{5}\b", re.I)
_MONEY = re.compile(r"\$\s?(\d+(?:\.\d{1,2})?)")


def _stem(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


def _tokens(text: str) -> set[str]:
    stop = {"the", "and", "for", "you", "that", "this", "with", "have", "are", "was", "but", "not",
            "your", "from", "about", "they", "what", "when", "any", "can", "just", "would", "like"}
    return {_stem(w) for w in _WORD.findall(text.lower()) if w not in stop and len(w) > 2}


def _focus(state: Any) -> str:
    """The customer's words, if the state has a field for them; else everything."""
    if isinstance(state, dict):
        for k in ("message", "customer_message", "query", "customer_request"):
            if k in state and isinstance(state[k], str):
                return state[k]
    return state if isinstance(state, str) else json.dumps(state)


def _hits(text: str, words: list[str]) -> int:
    t = text.lower()
    return sum(1 for w in words if w in t)


# ------------------------------------------------------------- confidence maths
def choice_confidence(probs: list[float]) -> float:
    n = len(probs)
    return 1.0 if n < 2 else max(0.0, (max(probs) - 1 / n) / (1 - 1 / n))


def score_confidence(probs: list[float]) -> float:
    n = len(probs)
    m = probs.index(max(probs))
    spread = sum(p * abs(i - m) for i, p in enumerate(probs))
    even = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1 - spread / even) if even else 1.0


class MockJev:
    def system_one(self, state: Any, questions: dict, **_: Any):
        answers = {name: self._answer(name, q, state) for name, q in questions.items()}
        return SimpleNamespace(model="mock-jev", answers=answers, usage=None)

    # ------------------------------------------------------------------ dispatch
    def _answer(self, name: str, q: dict, state: Any) -> dict:
        kind = q["type"]
        if kind == "choice":
            return self._choice(q, state)
        if kind == "score":
            return self._score(name, q, state)
        return {"type": "noul", "noul": round(self._noul(name, state, q.get("instructions", "")), 3)}

    # -------------------------------------------------------------------- choice
    def _choice(self, q: dict, state: Any) -> dict:
        text = _focus(state)
        toks = _tokens(text)
        # words that appear in EVERY option can't tell options apart, so drop them
        opt_toks = {k: _tokens(f"{k} {d or ''}") for k, d in q["criteria"].items()}
        shared = set.intersection(*opt_toks.values()) if len(opt_toks) > 1 else set()
        raw = {}
        for key in q["criteria"]:
            s = _hits(text, HINTS.get(key, []))
            if key not in HINTS:  # unknown option (sku, item id): match on its description
                s = len(toks & (opt_toks[key] - shared))
            raw[key] = (s + 0.1) ** 2
        total = sum(raw.values())
        probs = {k: round(v / total, 4) for k, v in raw.items()}
        best = max(probs, key=probs.get)
        return {"type": "choice", "choice": best, "probabilities": probs,
                "confidence": round(choice_confidence(list(probs.values())), 3)}

    # --------------------------------------------------------------------- score
    def _score(self, name: str, q: dict, state: Any) -> dict:
        n = len(q["criteria"])
        text = _focus(state)
        groups = sum(1 for k in ("order_status", "product_question", "recommendation",
                                 "return_refund", "complaint") if _hits(text, HINTS[k]))
        if name == "frustration":
            level = min(2, _hits(text, ANGER) + (1 if text.count("!") >= 2 else 0))
        elif name == "complexity":
            level = min(2, (groups >= 2) + 2 * bool(_hits(text, SAFETY + LEGAL))
                        + bool(re.search(r"\b(again|second|third)\b", text.lower())))
        elif name == "severity":
            level = min(3, 1 + bool(_hits(text, ANGER)) + 2 * bool(_hits(text, SAFETY)))
        elif name.startswith("fit_") and isinstance(state, dict):
            sku = name[4:]
            prod = state.get("candidates", {}).get(sku, {})
            prod_toks = _tokens(json.dumps(prod))
            level = min(n - 1, len(_tokens(text) & prod_toks))
        else:
            level = 0
        probs = [0.0] * n
        probs[level] = 0.8
        nbrs = [i for i in (level - 1, level + 1) if 0 <= i < n]
        for i in nbrs:
            probs[i] += 0.2 / len(nbrs)
        return {"type": "score", "score": round(sum(i * p for i, p in enumerate(probs)), 3),
                "probabilities": {str(i): round(p, 4) for i, p in enumerate(probs)},
                "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                "confidence": round(score_confidence(probs), 3)}

    # ---------------------------------------------------------------------- noul
    def _noul(self, name: str, state: Any, instructions: str = "") -> float:
        text = _focus(state)
        low = text.lower()
        groups = sum(1 for k in ("order_status", "product_question", "recommendation",
                                 "return_refund", "complaint") if _hits(text, HINTS[k]))
        rules = {
            "prompt_injection": lambda: 0.95 if _hits(text, INJECTION) else 0.02,
            "abusive": lambda: 0.9 if _hits(text, ABUSE) else 0.03,
            "off_topic": lambda: 0.05 if _hits(text, RETAIL) else (
                0.9 if re.search(r"capital of|weather|who is|president|poem|joke|recipe", low) else 0.35),
            "multi_request": lambda: 0.85 if groups >= 2 else 0.1,
            "wants_human": lambda: 0.9 if re.search(
                r"real person|human|representative|speak to (a|someone)|manager", low) else 0.05,
            "has_order_ref": lambda: 0.95 if _ORDER.search(text) else (0.6 if "order" in low else 0.1),
            "item_opened": lambda: 0.85 if re.search(r"opened|used|wore|worn|tried|after", low) else 0.2,
            "safety_issue": lambda: 0.92 if _hits(text, SAFETY) else 0.04,
            "legal_threat": lambda: 0.9 if _hits(text, LEGAL) else 0.03,
            "wants_compensation": lambda: 0.85 if re.search(r"refund|credit|compensat|money", low) else 0.2,
            "catalog_has_match": lambda: self._catalog_match(state),
            "grounded": lambda: self._grounded(state),
            "is_urgent": lambda: 0.9 if re.search(r"asap|urgent|tomorrow|need it|right now|today", low) else 0.15,
            "wants_refund": lambda: 0.88 if re.search(r"refund|money back", low) else 0.15,
        }
        if name in rules:
            return rules[name]()
        # generic fallback: do the statement's content words show up in the text?
        words = _tokens(instructions) - {"customer", "message", "item", "person", "would", "make",
                                         "good", "this", "statement"}
        return 0.8 if words & _tokens(text) else 0.2

    @staticmethod
    def _catalog_match(state: Any) -> float:
        if not isinstance(state, dict):
            return 0.3
        q = _tokens(_focus(state))
        best = max((len(q & _tokens(json.dumps(p))) for p in state.get("products", {}).values()),
                   default=0)
        return 0.9 if best >= 1 else 0.1

    @staticmethod
    def _grounded(state: Any) -> float:
        """Every $amount and order id in the reply must appear in the facts."""
        if not isinstance(state, dict):
            return 0.5
        reply, facts = state.get("reply", ""), json.dumps(state.get("facts", {}))
        fact_nums = {round(float(x), 2) for x in re.findall(r"\d+(?:\.\d+)?", facts)}
        for amt in _MONEY.findall(reply):
            if round(float(amt), 2) not in fact_nums:
                return 0.1
        for oid in _ORDER.findall(reply):
            if oid.upper() not in facts.upper():
                return 0.1
        return 0.95
