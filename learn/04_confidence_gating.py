"""Lesson 4: confidence is a second axis. The answer says WHAT, confidence says WHETHER to act.

Three bands is a good start:
    high   -> act automatically
    medium -> act, but confirm with the customer or log for review
    low    -> don't act; ask a question or hand to a human
Thresholds depend on what's at stake: a refund needs more confidence than showing a FAQ.
"""
from _setup import get_jev

INTENT = {"type": "choice", "instructions": "The customer's main request",
          "criteria": {"order_status": "Where is an order", "return_refund": "Return or refund",
                       "product_question": "Question about a product", "complaint": "Unhappy, wants it fixed"}}

STAKES = {"order_status": 0.5, "product_question": 0.5, "return_refund": 0.8, "complaint": 0.7}

messages = [
    "Where is order SS-10421?",
    "I'd like a refund for the jacket, it doesn't fit.",
    "hmm the thing from last week",
    "Can you check order SS-10455 and do you sell kettles?",
]

for m in messages:
    i = get_jev().ask({"message": m}, {"intent": INTENT})["intent"]
    need = STAKES.get(i.choice, 0.9)
    if i.confidence >= need:
        action = "ACT"
    elif i.confidence >= 0.4:
        action = "CONFIRM with customer"
    else:
        action = "ASK / HUMAN"
    print(f"{m[:55]:<56} {i.choice:<17} conf={i.confidence:.2f} need={need} -> {action}")
