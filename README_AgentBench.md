# AgentBench — Local LLM Inference Serving & Benchmarking

A learning-focused inference-serving project that runs a quantized **Qwen2.5-0.5B-Instruct** model locally with **MLX** on Apple silicon. It compares a sequential FastAPI inference endpoint with an experimental dynamic-batching endpoint and measures throughput and request latency.

> **Scope:** Local, single-machine experiments on an Apple M5 MacBook Pro (16 GB unified memory). Results are illustrative of this configuration and workload, not production benchmarks. MLX on Apple silicon uses Metal; this is **not** a CUDA/vLLM implementation.

## Architecture

```text
Concurrent HTTP clients / benchmark_real.py
                    |
                    v
            FastAPI /generate
                    |
          +---------+----------+
          |                    |
   real_server.py        batch_server.py
   Sequential baseline   Experimental batch server
   model_lock            asyncio request queue
   1 request per generate max batch size 8, wait up to 20 ms
          |                    |
      mlx_lm.generate()    mlx_lm.batch_generate()
          |                    |
          +---------+----------+
                    |
             Qwen2.5 / MLX
```

Both server variants are run **one at a time** on port `8081`.

## Setup

From the project directory:

```bash
cd ~/Downloads/agentbench_starter
source .venv/bin/activate
```

The project uses Python, FastAPI, Uvicorn, httpx, MLX, and mlx-lm. Install dependencies if setting up on a new machine; do not reinstall for an existing working virtual environment.

## Run the sequential baseline

**Terminal left:**

```bash
python -m uvicorn real_server:app --host 127.0.0.1 --port 8081
```

**Terminal right:**

```bash
python benchmark_real.py 1 20
python benchmark_real.py 2 20
python benchmark_real.py 5 20
```

Stop the server with `Control+C` before starting the batch server on the same port.

## Run the experimental dynamic-batching server

**Terminal left:**

```bash
python -m uvicorn batch_server:app --host 127.0.0.1 --port 8081
```

**Terminal right:**

```bash
curl -s http://127.0.0.1:8081/health
python benchmark_real.py 1 20
python benchmark_real.py 2 20
python benchmark_real.py 5 20
```

The batch server reported `max_batch_size: 8` and `max_tokens: 32` in its health response. In a 5-request concurrent test, each response reported `batch_size: 5`, confirming that requests were combined into a single batch.

## Measured results

### Online HTTP benchmark

Each row processes **20 requests**, with up to **32 generated tokens per response**. Results reflect one local run per configuration and may vary with warm-up and system load.

| Concurrency | Sequential throughput (req/s) | Batched throughput (req/s) | Sequential P95 (ms) | Batched P95 (ms) |
|---:|---:|---:|---:|---:|
| 1 | 8.07 | 6.33 | 137 | 181 |
| 2 | 8.49 | 12.34 | 245 | 193 |
| 5 | 8.34 | 21.82 | 617 | 283 |

At concurrency 5, the experimental dynamic-batching server delivered **2.62×** as many requests per second (21.82 vs 8.34) with approximately **54% lower P95 latency** (283 ms vs 617 ms) in these local test runs. At concurrency 1, batching was slower, in part because it waits up to 20 ms to collect requests.

### Request-level queueing investigation

Using the sequential server's model lock, five requests submitted together produced:

| Request | Lock wait (ms) | Generation (ms) | Server latency (ms) |
|---:|---:|---:|---:|
| 1 | 0 | 179 | 180 |
| 2 | 178 | 120 | 298 |
| 3 | 298 | 118 | 416 |
| 4 | 416 | 120 | 537 |
| 5 | 536 | 121 | 658 |

This exposed lock contention / serialized generation as a key source of queueing delay in the baseline.

### Offline fixed-batch scaling

`benchmark_batch_sizes.py` processed the **same 32 prompts** with a maximum of **32 output tokens each**, repeating each batch size **three times**. The figures below are reported medians.

| Batch size | Total time (s) | Output tokens/s | Speedup vs. batch 1 |
|---:|---:|---:|---:|
| 1 | 3.966 | 258.19 | 1.00× |
| 2 | 1.975 | 518.43 | 2.01× |
| 4 | 1.284 | 797.33 | 3.09× |
| 8 | 1.022 | 1001.55 | 3.88× |
| 16 | 0.946 | 1081.98 | 4.19× |
| 32 | 0.732 | 1399.56 | 5.42× |

The **5.42×** figure is from **offline fixed batching** and must not be presented as the HTTP service's speedup.

## Current limitations

- The dynamic batching server is an educational **fixed-batch** prototype, **not** production continuous batching or a vLLM/SGLang server.
- The test workload uses short prompts and 32-token completions; long and mixed workloads may behave differently.
- Online comparisons were not yet repeated across several independent runs or randomized order.
- Response-output token accounting is workload-/implementation-dependent; compare carefully across implementations.
- No full handling yet for request cancellation, bounded queues/backpressure, overload shedding, production observability, memory constraints, or comprehensive error handling.
- No TTFT/TPOT measurements or GPU memory profiling have been established.

## Next steps

1. Add reproducible benchmark runs with CSV output, warm-up policy, and repeated trials.
2. Test mixed prompt lengths, arrival rates, and batching window sizes.
3. Measure per-request TTFT, output token timing, and memory usage.
4. Add integration tests and bounded queue/backpressure to batch serving.
5. Compare local learning results with an NVIDIA/CUDA-based inference stack separately if desired.

## Files

| File | Purpose |
|---|---|
| `real_server.py` | Baseline HTTP inference with a model lock |
| `batch_server.py` | Experimental HTTP request queue and fixed-size dynamic batching |
| `benchmark_real.py` | Concurrent HTTP load test |
| `benchmark_batch_sizes.py` | Offline batch-size scaling experiment |
| `compare_batch.py` | Small sequential-vs-batched comparison |
| `real_inference.py` | Single local inference example |
| `app.py`, `benchmark.py`, `client.py` | Original mock-server exercises |

## Reproducibility notes

These are measurements from one developer laptop and should be reported with the exact device, model, prompt set, batch strategy, maximum generation length, and software versions. Keep baseline and batching results labeled separately.
