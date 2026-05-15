"""Tests for FastAPI health and basic endpoints."""
from fastapi.testclient import TestClient
from app.main import app


def test_health():
    """Test health check endpoint."""
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["backend"] == "NVIDIA NIM"


def test_chat_missing_messages():
    """Test chat endpoint rejects empty messages."""
    client = TestClient(app)
    r = client.post("/chat", json={"messages": []})
    assert r.status_code == 400


def test_ui_page_loads():
    """Test the UI page is served."""
    client = TestClient(app)
    r = client.get("/ui")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
