import asyncio
import math
import statistics
import sys
import time

import httpx

URL = "http://127.0.0.1:8081/generate"


async def run_one(client, semaphore, number):
    async with semaphore:
        start = time.perf_counter()

        response = await client.post(
            URL,
            json={
                "prompt": "Explain GPU inference in one sentence.",
                "max_tokens": 32
            }
        )

        response.raise_for_status()
        data = response.json()

        elapsed = time.perf_counter() - start

        return {
            "latency": elapsed,
            "tokens": data["generated_tokens"],
            "backend": data["backend"]
        }


async def main(concurrency, requests):
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=120) as client:
        start = time.perf_counter()

        results = await asyncio.gather(
            *[
                run_one(client, semaphore, i)
                for i in range(requests)
            ]
        )

        wall_time = time.perf_counter() - start

    latencies = sorted(
        result["latency"] for result in results
    )

    total_tokens = sum(
        result["tokens"] for result in results
    )

    p95_index = math.ceil(0.95 * len(latencies)) - 1
    p95 = latencies[p95_index]

    print(f"\nConcurrency: {concurrency}")
    print(f"Completed: {len(results)}/{requests}")
    print(f"Wall time: {wall_time:.3f}s")
    print(f"Throughput: {requests / wall_time:.2f} req/s")
    print(f"Output tokens: {total_tokens}")
    print(f"Output tokens/s: {total_tokens / wall_time:.2f}")
    print(f"Mean latency: {statistics.mean(latencies):.3f}s")
    print(f"P95 latency: {p95:.3f}s")
    print(f"Max latency: {max(latencies):.3f}s")
    print(f"Backend: {results[0]['backend']}")


if __name__ == "__main__":
    concurrency = int(sys.argv[1])
    requests = int(sys.argv[2])

    if concurrency < 1 or requests < 1:
        raise ValueError("Both values must be positive")

    asyncio.run(main(concurrency, requests))
