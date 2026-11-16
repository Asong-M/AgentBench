"""Simple concurrent request benchmark (end-to-end latency, not TTFT/TPOT)."""
import argparse
import asyncio
import statistics
import time
import httpx

async def run_request(client, idx):
    begin = time.perf_counter()
    try:
        r = await client.post("http://127.0.0.1:8080/generate", json={
            "prompt": f"Explain GPU to student {idx}", "max_tokens": 7
        })
        r.raise_for_status()
        return time.perf_counter() - begin, r.json()["generated_tokens"], None
    except Exception as exc:
        return time.perf_counter() - begin, 0, str(exc)

async def main(concurrency, requests):
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=180) as client:
        async def limited(i):
            async with sem:
                return await run_request(client, i)
        start = time.perf_counter()
        results = await asyncio.gather(*(limited(i) for i in range(requests)))
        duration = time.perf_counter() - start
    success = [r for r in results if r[2] is None]
    latency = sorted(r[0] for r in success)
    p95 = latency[max(0, -(-95 * len(latency) // 100) - 1)] if latency else 0
    print(f"Completed: {len(success)}/{requests}; wall time: {duration:.3f}s")
    print(f"Throughput: {len(success)/duration:.2f} req/s; generated tokens/s: {sum(r[1] for r in success)/duration:.2f}")
    if latency:
        print(f"Mean latency: {statistics.mean(latency):.3f}s; P95: {p95:.3f}s")
    if len(success) < requests:
        print("Errors (first 3):", [r[2] for r in results if r[2]][:3])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--requests", type=int, default=20)
    args = p.parse_args()
    asyncio.run(main(args.concurrency, args.requests))
