import rag


def test_system_prompt_contains_required_constraints() -> None:
    assert "strictly and only" in rag.SYSTEM_PROMPT
    assert "I don't know based on the provided Medium articles data." in rag.SYSTEM_PROMPT


def test_build_user_prompt_includes_question_and_passages() -> None:
    contexts = [
        {
            "article_id": "7",
            "title": "Habits 101",
            "chunk": "Start tiny and stay consistent.",
            "score": 0.9,
            "url": "http://x",
            "authors": "Dana",
        }
    ]
    prompt = rag.build_user_prompt("How do habits stick?", contexts)

    assert "How do habits stick?" in prompt
    assert "Habits 101" in prompt
    assert "article_id: 7" in prompt
    assert "Start tiny and stay consistent." in prompt


def test_build_user_prompt_handles_no_context() -> None:
    prompt = rag.build_user_prompt("anything?", [])
    assert "No article passages were retrieved." in prompt
    assert "anything?" in prompt


def test_answer_returns_required_schema(monkeypatch) -> None:
    fake_contexts = [
        {
            "article_id": "1",
            "title": "T1",
            "chunk": "c1",
            "score": 0.5,
            "url": "u1",
            "authors": "a1",
        }
    ]
    monkeypatch.setattr(rag, "retrieve", lambda q, top_k=10, namespace="": fake_contexts)
    monkeypatch.setattr(rag, "get_gpt5", lambda system, user: "final answer")

    result = rag.answer("a question")

    assert result["response"] == "final answer"
    assert result["context"] == [
        {"article_id": "1", "title": "T1", "chunk": "c1", "score": 0.5}
    ]
    assert set(result["Augmented_prompt"].keys()) == {"System", "User"}
    assert result["Augmented_prompt"]["System"] == rag.SYSTEM_PROMPT
    assert "a question" in result["Augmented_prompt"]["User"]
