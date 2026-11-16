import os
from fastapi.testclient import TestClient
from app import app

def test_health():
    os.environ["BACKEND"] = "mock"
    assert TestClient(app).get("/health").status_code == 200

def test_generate():
    os.environ["BACKEND"] = "mock"
    data = TestClient(app).post("/generate", json={"prompt": "hi", "max_tokens": 3}).json()
    assert data == {"text": "A GPU processes", "generated_tokens": 3, "backend": "mock"}
