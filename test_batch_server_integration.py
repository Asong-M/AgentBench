"""Integration tests for a running AgentBench batching server on port 8081.

Run with: python -m pytest -q test_batch_server_integration.py
"""
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

BASE_URL = "http://127.0.0.1:8081"


@pytest.fixture(scope="module", autouse=True)
def require_running_batch_server():
    try:
        response = httpx.get(f"{BASE_URL}/health", timeout=5)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        pytest.fail(
            "Start the batch server in the LEFT terminal first: "
            "python -m uvicorn batch_server:app --host 127.0.0.1 --port 8081\n"
            f"Connection error: {exc}"
        )
    assert response.json().get("backend") == "mlx-batch", (
        "Port 8081 is not running the batching backend."
    )


def test_health():
    data = httpx.get(f"{BASE_URL}/health", timeout=5).json()
    assert data["status"] == "ok"
    assert data["backend"] == "mlx-batch"
    assert 1 <= data["max_batch_size"]


def test_generate_response_fields():
    response = httpx.post(
        f"{BASE_URL}/generate",
        json={"prompt": "Define GPU in one sentence.", "max_tokens": 32},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    assert isinstance(data["text"], str)
    assert data["text"].strip()
    assert data["backend"] == "mlx-batch"
    assert 0 < data["generated_tokens"] <= 32
    assert 1 <= data["batch_size"] <= 8
    assert data["latency_seconds"] >= 0
    assert data["queue_wait_seconds"] >= 0
    assert data["generation_seconds"] >= 0


@pytest.mark.parametrize("bad_tokens", [0, -1, 9999])
def test_reject_invalid_max_tokens(bad_tokens):
    response = httpx.post(
        f"{BASE_URL}/generate",
        json={"prompt": "Hello", "max_tokens": bad_tokens},
        timeout=10,
    )
    assert response.status_code == 422


def test_five_concurrent_requests():
    def send_one(i):
        response = httpx.post(
            f"{BASE_URL}/generate",
            json={"prompt": f"Explain KV cache briefly. Case {i}.", "max_tokens": 32},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(send_one, range(5)))

    assert len(results) == 5
    for result in results:
        assert result["backend"] == "mlx-batch"
        assert isinstance(result["text"], str) and result["text"].strip()
        assert 1 <= result["batch_size"] <= 8
        assert 0 < result["generated_tokens"] <= 32
