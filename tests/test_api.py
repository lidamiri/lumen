import pytest
from fastapi.testclient import TestClient

from lumen.api.app import create_app


@pytest.fixture
def client(make_container):
    container = make_container()
    with TestClient(create_app(container)) as test_client:
        yield test_client


def test_health_reports_status(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "search_documents" in response.json()["tools"]


def test_chat_returns_completion(client):
    response = client.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "Echo: hi"
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]


def test_chat_rejects_invalid_role(client):
    response = client.post("/v1/chat", json={"messages": [{"role": "robot", "content": "hi"}]})
    assert response.status_code == 422


def test_ingest_ask_and_delete_flow(client):
    documents = [
        {"id": "guide.md", "text": "The warranty lasts twenty four months.", "metadata": {"team": "support"}}
    ]
    ingested = client.post("/v1/documents", json={"documents": documents})
    assert ingested.status_code == 200
    assert ingested.json()["chunks"] >= 1

    answer = client.post("/v1/ask", json={"question": "How long is the warranty?"})
    assert answer.status_code == 200
    assert answer.json()["grounded"] is True
    assert answer.json()["sources"][0]["document_id"] == "guide.md"

    removed = client.delete("/v1/documents/guide.md")
    assert removed.status_code == 200
    assert removed.json()["removed_chunks"] >= 1

    missing = client.delete("/v1/documents/guide.md")
    assert missing.status_code == 404


def test_ask_without_documents_is_not_grounded(client):
    response = client.post("/v1/ask", json={"question": "anything"})
    assert response.status_code == 200
    assert response.json()["grounded"] is False
    assert response.json()["sources"] == []


def test_agent_endpoint_answers_without_tools_for_echo_provider(client):
    response = client.post("/v1/agent", json={"question": "what is 2 + 3"})
    assert response.status_code == 200
    body = response.json()
    assert body["stop_reason"] == "answered"
    assert body["steps"] == []


def test_summary_and_classification_endpoints(client):
    summary = client.post("/v1/summaries", json={"text": "A long text about nothing in particular."})
    assert summary.status_code == 200
    assert summary.json()["summary"].startswith("Echo:")
    label = client.post("/v1/classifications", json={"text": "great", "labels": ["positive", "negative"]})
    assert label.status_code == 200
    assert "label" in label.json()


def test_oversized_input_is_rejected(make_container):
    container = make_container(max_input_chars=100)
    with TestClient(create_app(container)) as client:
        response = client.post("/v1/summaries", json={"text": "x" * 200})
        assert response.status_code == 422


def test_api_key_is_enforced_except_for_health(make_container):
    container = make_container(api_key="secret-key")
    with TestClient(create_app(container)) as client:
        assert client.get("/health").status_code == 200
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        assert client.post("/v1/chat", json=payload).status_code == 401
        assert client.post("/v1/chat", json=payload, headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.post("/v1/chat", json=payload, headers={"X-API-Key": "secret-key"}).status_code == 200
