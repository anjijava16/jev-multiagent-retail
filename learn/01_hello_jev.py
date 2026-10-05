"""Lesson 1: one state, one question, one number back.

Jev is not a chatbot. You give it:
    state     -> the thing to judge (text or JSON)
    questions -> named, typed questions about that state
and it gives back typed answers. Nothing to parse.
"""
import json

from _setup import get_jev

states = ["Hi, my package was supposed to arrive Monday and it's Thursday. I need it for a party tomorrow!",
         "Hi,Welcome"]
questions = {
    "is_urgent": {"type": "noul", "instructions": "The message conveys urgency or time pressure"},
}

# This is literally the JSON body that goes to POST https://api.typesafe.ai/v1/systemone
for state in states:
    print(f"STATE: {state}")
    print("REQUEST BODY\n", json.dumps({"state": state, "model": "jev-latest", "questions": questions}, indent=2))
    answers = get_jev().ask(state, questions)
    print(f"ANSWER BODY\n", answers)
    p = answers["is_urgent"].noul
    print(f"\nis_urgent = {p:.2f}   (probability the statement is TRUE, 0..1)")

    # Your code owns the decision. The model only gave you a calibrated number.
    if p >= 0.8:
        print("-> put this ticket at the top of the queue")
    else:
        print("-> normal queue")
    print("\n" + "="*50 + "\n")

# Same thing with the SDK's typed objects (requires `pip install typesafe-sdk` and a key):
#
#   from typesafe_sdk import TypeSafeClient, Noul
#   client = TypeSafeClient()                      # reads TYPESAFE_API_KEY
#   r = client.system_one(state=state, questions={"is_urgent": Noul(instructions="...")})
#   r.answers["is_urgent"].noul
