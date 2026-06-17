from fastapi.testclient import TestClient

import api.index as api_index
from api.index import app

client = TestClient(app)


def test_stats_returns_exact_schema() -> None:
    response = client.get("/api/stats")
    assert response.status_code == 200

    body = response.json()
    assert set(body.keys()) == {"chunk_size", "overlap_ratio", "top_k"}
    assert isinstance(body["chunk_size"], int)
    assert 0 <= body["overlap_ratio"] <= 0.3
    assert 1 <= body["top_k"] <= 30


def test_prompt_delegates_to_rag_answer(monkeypatch) -> None:
    captured = {}

    def fake_answer(question: str) -> dict:
        captured["question"] = question
        return {
            "response": "ok",
            "context": [],
            "Augmented_prompt": {"System": "s", "User": "u"},
        }

    monkeypatch.setattr(api_index, "answer", fake_answer)

    response = client.post("/api/prompt", json={"question": "what is X?"})

    assert response.status_code == 200
    assert captured["question"] == "what is X?"
    assert response.json()["response"] == "ok"


def test_prompt_requires_question_field() -> None:
    response = client.post("/api/prompt", json={})
    assert response.status_code == 422
