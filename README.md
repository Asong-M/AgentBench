# AgentBench — LLM Inference Serving & Performance Engineering

A learning-oriented, reproducible **local inference-serving prototype** built with Python, FastAPI, [MLX-LM](https://github.com/ml-explore/mlx-lm), and Qwen2.5-0.5B-Instruct-4bit on Apple Silicon (Apple M5, 16 GB unified memory).

> Scope: experimental local benchmark results, **not** a production SLO or a vLLM contribution. Fixed offline batching, online request coalescing, and continuous batching are different mechanisms.

## What it demonstrates

- A sequential HTTP LLM server with explicit `model_lock` and timing instrumentation.
- Offline batch-size scaling via `mlx_lm.batch_generate`.
- Online dynamic request **coalescing**: gather up to 8 requests for up to 20 ms, then execute one MLX batch.
- Bounded waiting queue (`maxsize=8`) and HTTP 503 admission control under burst overload.
- Client retries: no retry, fixed 200 ms, and capped exponential backoff with full jitter, up to 3 retries.
- Local integration tests and a lightweight GitHub Actions workflow for mock API tests.

## Measured highlights

| Experiment | Baseline | Treatment | Observation |
| --- | ---: | ---: | --- |
| Offline batch size (32 prompts, 32 output tokens each, 3 trials) | batch=1: 258.19 output tokens/s | batch=32: 1399.56 output tokens/s | **5.42×** higher offline throughput (median run) |
| Online HTTP serving (20 requests, concurrency=5) | sequential: 8.34 req/s, P95=617 ms | request coalescing: 21.82 req/s, P95=283 ms | **2.62×** throughput; P95 **54%** lower in these local runs |
| Queue cap burst (100 concurrent requests, one run) | — | 16 HTTP 200, 84 HTTP 503 | Bounded admission under overload |
| Retry strategies (50 jobs per policy, 3 sequential trials) | no retry: 16/50 success | exponential + jitter: 50/50 success | Higher completion at greater end-to-end latency and attempt cost |

See [Experiment log](docs/EXPERIMENTS.md) for **all** measurements, limitations, and interpretation.

## Architecture

```
HTTP clients
  |
  +--> /health
  |
  +--> /generate --> FastAPI validation
                        |
                 Admission control --> HTTP 503 if queue full
                        |
                  bounded asyncio.Queue
                        |
               batch worker (up to 8 / 20 ms)
                        |
                   asyncio.to_thread
                        |
                MLX-LM batch_generate --> individual HTTP 200 results
```

The sequential baseline (`real_server.py`) instead serializes each inference under a thread lock. The offline batching harness bypasses HTTP. The online batch worker is **not** token-step continuous batching.

## Files / getting started

- `real_server.py`: sequential real-model server.
- `batch_server.py`: online batch coalescing.
- `batch_server_backpressure.py`: online coalescing + bounded queue and 503.
- `benchmark_real.py`: end-to-end HTTP load test.
- `benchmark_batch_sizes.py`: offline batch-size sweep.
- `benchmark_retry.py`, `compare_retry_policies.py`: client retry experiments.
- `test_batch_server_integration.py`: **local** live-server integration tests.
- `test_app.py`: mock API test suitable for general CI.

This MLX version targets supported Apple Silicon macOS environments; GPU-related results are **not** directly comparable with NVIDIA/CUDA serving engines.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install mlx-lm

# Terminal A (choose exactly one server, port 8081)
python -m uvicorn batch_server_backpressure:app --host 127.0.0.1 --port 8081

# Terminal B
curl -s http://127.0.0.1:8081/health
python -m pytest -v test_batch_server_integration.py
python benchmark_real.py 5 20
python benchmark_batch_sizes.py
python compare_retry_policies.py
```

**Configuration constraint:** The bounded batch server used in these experiments requires `max_tokens=32`; other values receive HTTP 422. The examples and live integration tests intentionally use 32. `requirements.txt` in older repository versions may omit `mlx-lm`: install it separately as shown, or add it as a platform-specific dependency before publishing a new setup guide.

## Engineering documentation

- [Full experiment history and caveats](docs/EXPERIMENTS.md)
- [Design decisions and failure modes](docs/ARCHITECTURE_AND_TRADEOFFS.md)
- [Optimization roadmap](docs/ROADMAP.md)

## Important limitations

- Small model, short fixed output length, single local machine, limited trial counts.
- Benchmarks were conducted at different times; they are **not** controlled simultaneous A/B experiments.
- The measured P95 for retry policies **includes successful and failed jobs**. Short failure latency must not be confused with better user experience.
- Successful 503 rejection does **not** establish bounded total model memory, fairness, cancellation safety, or production readiness.
- Retry is only safe for appropriate operations; a timeout with unknown outcome needs careful idempotency semantics.
- No production monitoring, autoscaling, multi-replica deployment, or vLLM backend in the current implementation.

## Public documentation only

This repository intentionally excludes personal job applications, résumé drafts, private interview preparation, and personal identifiers.
