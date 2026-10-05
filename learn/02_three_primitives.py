"""Lesson 2: Choice, Score and Noul in ONE call.

All questions are evaluated in parallel, each in isolation, against the same
state. Adding questions barely changes latency, and they don't contaminate
each other (no "context rot" like a long LLM prompt).
"""
from _setup import get_jev

state = "I ordered the merino sweater in M but got an L. Can I swap it? Bit annoying, it was a gift."

questions = {
    # CHOICE: pick one option from a dict {key: description}
    "intent": {"type": "choice", "instructions": "What the customer wants",
               "criteria": {"order_status": "Where is my order",
                            "return_refund": "Return, exchange or refund",
                            "product_question": "Question about a product"}},
    # SCORE: place the state on an ORDERED list of levels (index 0 = lowest)
    "frustration": {"type": "score", "instructions": "How frustrated the customer sounds",
                    "criteria": ["Calm", "Annoyed but civil", "Very angry"]},
    # NOUL: probability that a statement is true
    "is_gift": {"type": "noul", "instructions": "The item was bought as a gift"},
}

a = get_jev().ask(state, questions)

i = a["intent"]
print(f"CHOICE intent      -> {i.choice}  confidence={i.confidence:.2f}")
for opt, p in i.top():
    print(f"      {opt:<17} {p:.2f}")

f = a["frustration"]
print(f"\nSCORE frustration  -> {f.score:.2f} (expected level; 0=Calm, 2=Very angry) "
      f"confidence={f.confidence:.2f}")
print("      probabilities per level:", f.probabilities)

print(f"\nNOUL is_gift       -> {a['is_gift'].noul:.2f}")

print("""
What to remember
  .choice / .score / .noul   the answer
  .probabilities             the full distribution (Choice and Score)
  .confidence                0 = spread evenly, 1 = all on one answer (Choice and Score)
  Noul has no .confidence; distance from 0.5 IS the confidence: |2p - 1|
""")
