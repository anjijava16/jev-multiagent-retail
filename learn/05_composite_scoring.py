"""Lesson 5: don't ask "how good is this product for them?" in one go.

Split it into atomic Scores (each a quick gut check), get them all in ONE call,
then combine them with weights that live in your code. When the business
changes its mind, you change a number, not a prompt.
"""
from _setup import get_jev

LEVELS = ["Poor", "OK", "Good", "Excellent"]
request = "Gift for my dad who loves coffee, around $80"
products = {
    "HM-410": "Brewline Pour-Over Coffee Kit, $45, dripper + grinder, popular gift",
    "HM-420": "Kettle Pro Gooseneck Kettle, $69, for pour-over coffee and tea",
    "AP-520": "Merino Everyday Crew Sweater, $89",
}

questions = {}
for sku, desc in products.items():
    questions[f"fit_{sku}"] = {"type": "score", "instructions": f"How well {sku} matches the request", "criteria": LEVELS}
    questions[f"giftable_{sku}"] = {"type": "noul", "instructions": f"{sku} would make a good gift for this person"}

state = {"customer_request": request, "candidates": {k: {"description": v} for k, v in products.items()}}
a = get_jev().ask(state, questions)   # 6 questions, 1 request

W_FIT, W_GIFT = 0.7, 0.3              # <- the knobs you own
rows = []
for sku in products:
    fit = a[f"fit_{sku}"].score / 3
    gift = a[f"giftable_{sku}"].noul
    rows.append((W_FIT * fit + W_GIFT * gift, sku, fit, gift))
for total, sku, fit, gift in sorted(rows, reverse=True):
    print(f"{sku}  composite={total:.2f}  fit={fit:.2f}  giftable={gift:.2f}")
