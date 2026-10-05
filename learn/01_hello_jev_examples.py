"""Lesson 1b: the same Noul on messages where the answer should be TRUE and FALSE."""
from _setup import get_jev

# questions = {
#     "is_urgent": {"type": "noul", "instructions": "The message conveys urgency or time pressure"},
# }

questions = {
    "is_urgent": {"type": "noul", "instructions": "The message conveys IPL Cricket Info"},
}

examples = [
    # expected TRUE (high probability)
    ("TRUE",  "Hi, my package was supposed to arrive Monday and it's Thursday. I need it for a party tomorrow!"),
    ("TRUE",  "Please help ASAP, I was charged twice and my rent is due today."),
    # expected FALSE (low probability)
    ("FALSE", "Hi, welcome"),
    ("FALSE", "Just browsing, no rush at all. I might order something next month."),
    ("FALSE", "Thanks, the jacket arrived and fits well."),
    ("FALSE", "Whenever you get a chance, could you tell me what colours the sweater comes in?"),
]

for expected, state in examples:
    p = get_jev().ask(state, questions)["is_urgent"].noul
    got = p >= 0.5                      # your threshold turns probability into True/False
    print(f"got={str(got)}, expected={expected:<5}")
    print(f"expected={expected:<5} noul={p:.2f} -> {str(got):<5} | {state}")