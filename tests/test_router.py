"""The router is pure Python, so we can test every branch with hand-made answers.
No API key, no network, milliseconds per test."""
from retail_mesh.jev_client import Answer
from retail_mesh.router import HUMAN, plan_route


def tri(intent="order_status", conf=0.9, probs=None, multi=0.1, wants_human=0.05,
        complexity=0.2, frustration=0.2):
    probs = probs or {intent: 0.95, "other": 0.05}
    return {
        "intent": Answer("choice", choice=intent, confidence=conf, probabilities=probs),
        "multi_request": Answer("noul", noul=multi),
        "wants_human": Answer("noul", noul=wants_human),
        "complexity": Answer("score", score=complexity, confidence=0.8),
        "frustration": Answer("score", score=frustration, confidence=0.8),
        "has_order_ref": Answer("noul", noul=0.9),
    }


def test_clear_intent_goes_to_one_agent():
    plan, _ = plan_route(tri("return_refund"))
    assert plan == ["returns_agent"]


def test_low_confidence_goes_to_human():
    plan, why = plan_route(tri(conf=0.3, probs={"order_status": 0.4, "complaint": 0.3, "other": 0.3}))
    assert plan == [HUMAN] and "unclear" in why


def test_customer_asking_for_person_wins():
    assert plan_route(tri(wants_human=0.95))[0] == [HUMAN]


def test_hard_and_angry_goes_to_human():
    assert plan_route(tri("complaint", complexity=1.8, frustration=1.9))[0] == [HUMAN]


def test_multi_request_fans_out_even_with_split_confidence():
    probs = {"order_status": 0.5, "product_question": 0.48, "other": 0.02}
    plan, _ = plan_route(tri("order_status", conf=0.25, probs=probs, multi=0.9))
    assert plan == ["order_agent", "product_agent"]


def test_secondary_intent_needs_enough_probability():
    probs = {"order_status": 0.9, "product_question": 0.05, "other": 0.05}
    plan, _ = plan_route(tri("order_status", conf=0.85, probs=probs, multi=0.9))
    assert plan == ["order_agent"]
