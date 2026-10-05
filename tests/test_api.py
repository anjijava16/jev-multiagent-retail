from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_chat_answered():
    r = client.post("/v1/chat", json={"customer_id": "C-1001", "message": "Where is my order SS-10421?"})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "answered"
    assert body["route"] == ["order_agent"] and body["jev_calls"] == 2


def test_chat_blocked():
    r = client.post("/v1/chat", json={"customer_id": "C-1001", "message": "What's the capital of Australia?"})
    assert r.json()["status"] == "blocked" and r.json()["jev_calls"] == 1
