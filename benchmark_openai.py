"""AgentBench: benchmark an OpenAI-compatible non-streaming Chat Completions API.

Examples:
    python benchmark_openai.py --concurrency 5 --requests 20
    python benchmark_openai.py --base-url http://127.0.0.1:8000 \
        --model your-served-model --concurrency 5 --requests 20

The client measures end-to-end HTTP request duration, not server-side TTFT.
It does not retry 503 responses; this makes overload behavior visible.
"""

import argparse
import asyncio
import math
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import httpx

DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


@dataclass
class Result:
    status: str
    latency_seconds: float
    completion_tokens: Optional[int] = None
    batch_size: Optional[int] = None
    error: Optional[str] = None


def percentile(values: list[float], pct: float) -> Optional[float]:
    """Nearest-rank percentile; returns None for an empty sample."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return ordered[rank - 1]


def fmt_seconds(value: Optional[float]) -> str:
    return f"{value:.3f}s" if value is not None else "N/A"


async def one_request(
    client: httpx.AsyncClient, semaphore: asyncio.Semaphore, url: str,
    model: str, prompt: str, max_tokens: int,
) -> Result:
    async with semaphore:
        started = time.perf_counter()
        try:
            response = await client.post(
                url,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "stream": False,
                },
            )
            duration = time.perf_counter() - started
            status = str(response.status_code)
            if response.status_code != 200:
                return Result(status=status, latency_seconds=duration)
            try:
                body = response.json()
            except ValueError:
                return Result(status="invalid_json", latency_seconds=duration,
                              error="HTTP 200 response was not valid JSON")
            if not isinstance(body, dict) or not isinstance(body.get("choices"), list):
                return Result(status="invalid_schema", latency_seconds=duration,
                              error="HTTP 200 response had no choices array")

            usage = body.get("usage") or {}
            metrics = body.get("agentbench_metrics") or {}
            tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
            batch = metrics.get("batch_size") if isinstance(metrics, dict) else None
            return Result(
                status=status,
                latency_seconds=duration,
                completion_tokens=tokens if type(tokens) is int and tokens >= 0 else None,
                batch_size=batch if type(batch) is int and batch > 0 else None,
            )
        except httpx.RequestError as exc:
            return Result(status="network_error", latency_seconds=time.perf_counter() - started,
                          error=f"{type(exc).__name__}: {exc}")


async def benchmark(args: argparse.Namespace) -> None:
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    print("AgentBench OpenAI-compatible Chat Completions Benchmark")
    print(f"URL: {url}")
    print(f"Model: {args.model}")
    print(f"Requests: {args.requests}; concurrency: {args.concurrency}; max_tokens: {args.max_tokens}")
    print("Retries: disabled; streaming: disabled")

    limits = httpx.Limits(
        max_connections=max(args.concurrency, 10),
        max_keepalive_connections=max(args.concurrency, 10),
    )
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout), limits=limits) as client:
        semaphore = asyncio.Semaphore(args.concurrency)
        started = time.perf_counter()
        results = await asyncio.gather(*[
            one_request(client, semaphore, url, args.model, args.prompt, args.max_tokens)
            for _ in range(args.requests)
        ])
        wall = time.perf_counter() - started

    counts = Counter(r.status for r in results)
    successful = [r for r in results if r.status == "200"]
    successful_latencies = [r.latency_seconds for r in successful]
    all_latencies = [r.latency_seconds for r in results]
    known_tokens = [r.completion_tokens for r in successful if r.completion_tokens is not None]
    batch_sizes = Counter(r.batch_size for r in successful if r.batch_size is not None)

    print("\n=== Results ===")
    print(f"Status counts: {dict(sorted(counts.items()))}")
    print(f"Successful jobs: {len(successful)}/{args.requests} ({len(successful)/args.requests:.1%})")
    print(f"Wall time: {wall:.3f}s")
    print(f"Completed successes / wall time: {len(successful)/wall:.2f} req/s")
    print(f"P50 latency (all outcomes): {fmt_seconds(percentile(all_latencies, 50))}")
    print(f"P95 latency (all outcomes): {fmt_seconds(percentile(all_latencies, 95))}")
    print(f"P50 latency (HTTP 200 only): {fmt_seconds(percentile(successful_latencies, 50))}")
    print(f"P95 latency (HTTP 200 only): {fmt_seconds(percentile(successful_latencies, 95))}")
    if known_tokens:
        print(f"Output tokens reported: {sum(known_tokens)} across {len(known_tokens)} successful responses")
        if len(known_tokens) == len(successful):
            print(f"Aggregate output tokens / wall time: {sum(known_tokens)/wall:.2f} tokens/s")
        else:
            print("Aggregate tokens/s: N/A (some successful responses omitted completion_tokens)")
    else:
        print("Aggregate tokens/s: N/A (no completion_tokens reported)")
    if batch_sizes:
        print(f"Reported AgentBench batch sizes (per response): {dict(sorted(batch_sizes.items()))}")
    errors = [r.error for r in results if r.error]
    if errors:
        print(f"Example error: {errors[0]}")
    print("\nNotes: HTTP 503 counts as a failed job, not a completed inference.")
    print("P50/P95 are client-side end-to-end latency, NOT TTFT; non-streaming responses cannot measure TTFT.")
    print("Compare backends only with compatible workloads/hardware assumptions.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prompt", default="Explain inference batching in one sentence.")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1 or args.max_tokens < 1 or args.timeout <= 0:
        parser.error("requests, concurrency, max-tokens and timeout must all be positive")
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
