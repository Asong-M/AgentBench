"""AgentBench: compare burst traffic with and without bounded 503 retries.

Run while batch_server_backpressure is listening on http://127.0.0.1:8081:
    python benchmark_retry.py

No changes to the server are required. Requires httpx.
"""

import asyncio
import math
import random
import statistics
import time
from collections import Counter

import httpx

URL = "http://127.0.0.1:8081/generate"
REQUESTS = 50
MAX_RETRIES = 3  # Three retries after the initial attempt (at most four attempts)
BASE_DELAY_SECONDS = 0.25
MAX_DELAY_SECONDS = 4.0
JITTER_MAX_SECONDS = 0.25


def retry_delay(retry_number: int, response: httpx.Response) -> float:
    """Honor Retry-After, back off exponentially, add jitter."""
    exponential = min(MAX_DELAY_SECONDS, BASE_DELAY_SECONDS * (2 ** (retry_number - 1)))
    header = response.headers.get("retry-after")
    try:
        server_minimum = max(0.0, float(header)) if header is not None else 0.0
    except ValueError:
        # This local prototype sends a numeric Retry-After, not an HTTP-date.
        server_minimum = 0.0
    return max(exponential, server_minimum) + random.uniform(0, JITTER_MAX_SECONDS)


async def send_one(client: httpx.AsyncClient, request_number: int, allow_retries: bool):
    start = time.perf_counter()
    counts = Counter()
    final_status = "unknown"
    attempts = 1 + (MAX_RETRIES if allow_retries else 0)

    for attempt in range(attempts):
        try:
            response = await client.post(
                URL,
                json={"prompt": f"Explain GPU inference in one sentence. Job {request_number}.", "max_tokens": 32},
            )
            final_status = response.status_code
            counts[response.status_code] += 1
        except httpx.RequestError:
            final_status = "network_error"
            counts["network_error"] += 1
            break

        if final_status != 503 or attempt == attempts - 1:
            break

        await asyncio.sleep(retry_delay(attempt + 1, response))

    return {
        "status": final_status,
        "attempts": sum(counts.values()),
        "duration": time.perf_counter() - start,
        "attempt_statuses": counts,
    }


async def run_case(label: str, allow_retries: bool):
    random.seed(42)
    async with httpx.AsyncClient(timeout=120) as client:
        start = time.perf_counter()
        results = await asyncio.gather(
            *(send_one(client, i, allow_retries) for i in range(REQUESTS))
        )
        elapsed = time.perf_counter() - start

    final = Counter(r["status"] for r in results)
    attempts = Counter()
    for r in results:
        attempts.update(r["attempt_statuses"])
    durations = sorted(r["duration"] for r in results)
    p95 = durations[math.ceil(0.95 * len(durations)) - 1]
    print(f"\n=== {label} ===")
    print(f"Jobs: {REQUESTS}")
    print(f"Successful jobs (final 200): {final[200]}")
    print(f"Failed jobs (final 503): {final[503]}")
    print(f"Network errors: {final['network_error']}")
    print(f"Total HTTP attempts: {sum(v for k, v in attempts.items() if isinstance(k, int))}")
    print(f"HTTP 503 responses across all attempts: {attempts[503]}")
    print(f"Wall time: {elapsed:.3f}s")
    print(f"Mean end-to-end job latency: {statistics.mean(durations):.3f}s")
    print(f"P95 end-to-end job latency: {p95:.3f}s")
    return final


async def main():
    print("AgentBench 503 Retry / Backpressure Experiment")
    print(f"Burst size: {REQUESTS}; max retries: {MAX_RETRIES}")
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get("http://127.0.0.1:8081/health")
        response.raise_for_status()
        health = response.json()
        if "max_queue_size" not in health:
            raise RuntimeError("Expected the backpressure server on port 8081; found a different server.")
        print("Backpressure health:", health)
    await run_case("Without retries", allow_retries=False)
    await asyncio.sleep(1.5)  # Let the server finish the previous burst.
    await run_case("Bounded exponential backoff + jitter", allow_retries=True)


if __name__ == "__main__":
    asyncio.run(main())
