"""AgentBench: compare overload retry policies against the running HTTP server.

Run: python compare_retry_policies.py
Requires: httpx; batch_server_backpressure running on localhost:8081.
"""
import asyncio
import math
import random
import statistics
import time
from collections import Counter

import httpx

BASE_URL = "http://127.0.0.1:8081"
JOBS = 50
REPEATS = 3
MAX_RETRIES = 3
FIXED_DELAY_SECONDS = 0.2
BASE_BACKOFF_SECONDS = 0.2
MAX_BACKOFF_SECONDS = 1.6
COOLDOWN_SECONDS = 2
POLICIES = ("no_retry", "fixed_200ms", "exponential_jitter")


def percentile_95(values):
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def next_delay(policy, retry_number, response, rng):
    """retry_number starts at 1. Always apply jitter for exponential policy."""
    if policy == "fixed_200ms":
        return FIXED_DELAY_SECONDS
    if policy == "exponential_jitter":
        cap = min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * (2 ** (retry_number - 1)))
        delay = rng.uniform(0, cap)  # full jitter
        # Optional Retry-After from server; respect if a numeric seconds value.
        header = response.headers.get("retry-after")
        if header:
            try:
                delay = max(delay, min(10.0, max(0.0, float(header))))
            except ValueError:
                pass
        return delay
    raise ValueError(policy)


async def one_job(client, policy, index, rng):
    start = time.perf_counter()
    attempts = 0
    rejected = 0
    status = "network_error"

    for retry_number in range(MAX_RETRIES + 1):
        attempts += 1
        try:
            response = await client.post(
                f"{BASE_URL}/generate",
                json={
                    "prompt": f"Explain GPU inference in one sentence. Case {index}.",
                    "max_tokens": 32,
                },
            )
            status = response.status_code
        except httpx.RequestError:
            status = "network_error"
            break  # do not retry unknown network errors: may have completed

        if status == 200:
            break
        if status != 503 or policy == "no_retry" or retry_number == MAX_RETRIES:
            break

        rejected += 1
        await asyncio.sleep(next_delay(policy, retry_number + 1, response, rng))

    # Count final 503 as well.
    if status == 503:
        rejected += 1
    return {
        "status": status,
        "attempts": attempts,
        "http_503": rejected,
        "latency": time.perf_counter() - start,
    }


async def run_trial(policy, trial_number):
    async with httpx.AsyncClient(timeout=60) as client:
        start = time.perf_counter()
        results = await asyncio.gather(*[
            one_job(
                client,
                policy,
                i,
                random.Random(2026 + trial_number * 1000 + i),
            )
            for i in range(JOBS)
        ])
        wall_time = time.perf_counter() - start

    counts = Counter(r["status"] for r in results)
    return {
        "success": counts[200],
        "failed_503": counts[503],
        "other": JOBS - counts[200] - counts[503],
        "attempts": sum(r["attempts"] for r in results),
        "responses_503": sum(r["http_503"] for r in results),
        "wall": wall_time,
        "mean_latency": statistics.mean(r["latency"] for r in results),
        "p95": percentile_95([r["latency"] for r in results]),
    }


async def main():
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            response = await client.get(f"{BASE_URL}/health")
            response.raise_for_status()
            health = response.json()
        except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as exc:
            print(f"Cannot reach server at {BASE_URL}: {exc}")
            print("Start batch_server_backpressure on port 8081 in the left iTerm2 pane.")
            return
    if health.get("max_queue_size") is None:
        print("Expected backpressure server. Actual health response:", health)
        return

    print("Server health:", health)
    print(f"Jobs/trial={JOBS}; trials/policy={REPEATS}; maximum retries={MAX_RETRIES}")
    print("P95 includes both successful and unsuccessful jobs; report success alongside latency.")
    print("These are sequential trials, not identical simultaneous A/B load tests.\n")

    samples = {p: [] for p in POLICIES}
    # Rotate policy order across trials to reduce systematic warm-up bias.
    for trial in range(REPEATS):
        order = POLICIES[trial:] + POLICIES[:trial]
        for policy in order:
            result = await run_trial(policy, trial)
            samples[policy].append(result)
            print(
                f"Trial {trial+1} {policy:20s} "
                f"ok={result['success']:2d}/{JOBS} "
                f"503={result['responses_503']:3d} "
                f"attempts={result['attempts']:3d} "
                f"wall={result['wall']:.2f}s "
                f"p95={result['p95']:.2f}s"
            )
            await asyncio.sleep(COOLDOWN_SECONDS)

    print("\n=== MEDIAN ACROSS THREE TRIALS ===")
    print(f"{'Policy':20s} {'Success':>8s} {'Attempts':>9s} {'503':>6s} {'Wall(s)':>9s} {'P95(s)':>8s}")
    for policy in POLICIES:
        rows = samples[policy]
        med = lambda key: statistics.median(r[key] for r in rows)
        print(
            f"{policy:20s} {med('success'):>5.0f}/{JOBS:<2d} "
            f"{med('attempts'):>9.0f} {med('responses_503'):>6.0f} "
            f"{med('wall'):>9.2f} {med('p95'):>8.2f}"
        )
    print("\nFinished. Compare success, attempts, and P95 together—not latency alone.")


if __name__ == "__main__":
    asyncio.run(main())
