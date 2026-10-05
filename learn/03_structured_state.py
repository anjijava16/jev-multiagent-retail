"""Lesson 3: state can be a JSON object. Put everything a judge would need in it.

Rule of thumb: facts go in the state, judgments go in the questions,
arithmetic stays in Python.
"""
from datetime import date

from _setup import get_jev

state = {
    "message": "The headphones crackle in the left ear since day one. I want my money back.",
    "order": {"id": "SS-10421", "delivered": "2026-09-20", "item": "Aria Pro Headphones", "price": 249.0},
    "policy": "Defective items can be returned within 90 days. Other returns: 30 days.",
}

a = get_jev().ask(state, {
    "reason": {"type": "choice", "instructions": "Why the customer wants to return the item",
               "criteria": {"defective": "Fault or malfunction", "size_fit": "Wrong size",
                            "changed_mind": "No longer wants it"}},
    "wants_refund": {"type": "noul", "instructions": "The customer asks for money back rather than a replacement"},
})

# Dates: code, not model. Never ask a model "is this within 90 days?"
days = (date(2026, 10, 4) - date.fromisoformat(state["order"]["delivered"])).days
window = 90 if a["reason"].choice == "defective" else 30

print(f"reason={a['reason'].choice} ({a['reason'].confidence:.2f})  wants_refund={a['wants_refund'].noul:.2f}")
print(f"{days} days since delivery, window {window} -> eligible={days <= window}")
