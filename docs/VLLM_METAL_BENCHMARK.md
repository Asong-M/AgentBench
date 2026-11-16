# AgentBench: Initial MLX vs vLLM-Metal Benchmark

**Experiment date:** 2026-11-15 (US Pacific)  
**Status:** Exploratory local measurements; not a controlled production benchmark.

## Purpose

Evaluate whether the same OpenAI-compatible benchmark client can invoke both:

- **AgentBench MLX backend:** custom FastAPI serving with bounded queue, fixed-window dynamic batching and an OpenAI-style adapter (`openai_adapter.py`).
- **vLLM-Metal backend:** `vllm serve` with the vLLM-Metal plugin on Apple Silicon.

The goal is to characterize throughput/latency behavior and establish a repeatable testing workflow, **not** to claim that either backend is universally faster.

## Environment and method

- Device: Apple M5 MacBook Pro, 16 GB memory, Apple Silicon (`arm64`).
- macOS: 26.5.1.
- vLLM CLI reports `0.33.0+cpu`; installation reports `vllm-metal==0.33.0`. The `+cpu` version suffix alone does not identify whether GPU acceleration is active.
- vLLM Python environment: Python 3.12.10 (`~/.venv-vllm-metal`). AgentBench project environment: Python 3.13.12.
- Model identifier: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`.
- API endpoint: `/v1/chat/completions`; vLLM server at `http://127.0.0.1:8000` and MLX adapter previously at `http://127.0.0.1:8081`.
- Benchmark client: `benchmark_openai.py` (non-streaming, no retries).
- `max_tokens=32`; the test results below were obtained at the indicated concurrency and number of total requests.
- Throughput = successfully completed requests / wall-clock test duration. Output token throughput = sum of API-reported completion tokens / wall-clock duration.
- P50/P95 = **client-observed end-to-end request latency**, not time-to-first-token (TTFT).

## MLX adapter: initial baseline

One local run of 20 requests at concurrency 5:

| Requests | Concurrency | HTTP 200 | Wall time | Throughput (req/s) | P50 | P95 | Reported output tokens | Output tokens/s | Reported batch sizes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 20 | 5 | 20/20 | 0.916 s | 21.82 | 0.204 s | 0.283 s | 640 | 698.38 | `{5: 20}` |

This confirms the adapter supported the workload and the server reported batches of five. It is **not** yet a matched, repeated baseline for the vLLM concurrency sweep.

## vLLM-Metal: exploratory runs

First and second runs at concurrency 5 illustrate run-to-run variation:

| Run | Requests | Concurrency | HTTP 200 | Wall time | Throughput (req/s) | P50 | P95 | Output tokens | Output tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Initial | 20 | 5 | 20/20 | 1.610 s | 12.42 | 0.235 s | 0.921 s | 631 | 391.81 |
| Repeat | 20 | 5 | 20/20 | 1.230 s | 16.27 | 0.188 s | 0.646 s | 640 | 520.49 |

Subsequent concurrency sweep (the **repeat** above is the concurrency-5 row):

| Requests | Concurrency | HTTP 200 | Wall time | Throughput (req/s) | P50 | P95 | Output tokens | Output tokens/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 5 | 20/20 | 1.230 s | 16.27 | 0.188 s | 0.646 s | 640 | 520.49 |
| 40 | 10 | 40/40 | 1.906 s | 20.99 | 0.340 s | 0.866 s | 1,280 | 671.73 |
| 80 | 20 | 80/80 | 3.132 s | 25.55 | 0.652 s | 1.218 s | 2,514 | 802.78 |
| 160 | 40 | 160/160 | 4.581 s | 34.93 | 1.016 s | 1.524 s | 5,036 | 1,099.27 |

All requests in these runs returned HTTP 200. There were no recorded HTTP 503 responses at the tested vLLM-Metal load levels.

### Observations

1. Across the tested vLLM-Metal configurations, higher concurrency coincided with higher aggregate throughput **and** higher P95 latency.
2. At concurrency 40, the maximum **observed** throughput was 34.93 completed requests/s and 1,099.27 reported output tokens/s; the P95 was 1.524 s. This does **not** establish saturation or maximum achievable throughput.
3. The repeated concurrency-5 run was faster than the initial run. Warm-up is one possible factor, but the measurement does not establish causality.
4. The one-run MLX baseline at concurrency 5 (21.82 req/s, P95 0.283 s) differs from the second vLLM run (16.27 req/s, P95 0.646 s). This is an **exploratory comparison only**, not a conclusive efficiency ranking.

## Important limitations and fairness considerations

- **Not a controlled A/B test:** the backends were tested sequentially with limited repeated measurements and potentially different warm-up/cache/thermal conditions.
- **Request count varies with concurrency:** 20, 40, 80 and 160 requests respectively, so workload duration also varies. Use equal request counts or steady-state duration when comparing load levels systematically.
- **Token counts can differ:** e.g., 2,514 or 5,036 tokens instead of exactly 32 × number of requests. The APIs may differ in stop behavior, tokenization or prompt/chat formatting; actual prompts and generation settings should be pinned and logged.
- **Different implementations:** the custom MLX server uses fixed-window batching; vLLM-Metal uses a different scheduler and may have different resource and configuration limits. API compatibility does not imply identical execution semantics.
- **Hardware/backend verification:** independently verify which compute device and attention kernels are being used before making statements about GPU utilization or engine-level performance.
- **Not TTFT or TPOT:** non-streaming end-to-end latency and aggregate tokens/s do not measure first-token latency or time per output token. Implement streaming and record event timestamps for those metrics.
- **No production claims:** short local tests do not establish sustained service capacity, operational reliability, or suitability for production traffic.

## Reproduction (vLLM-Metal)

The vLLM-Metal environment is separate from the AgentBench `.venv`. For example, with a compatible vLLM-Metal installation:

```bash
source ~/.venv-vllm-metal/bin/activate
vllm serve mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --host 127.0.0.1 --port 8000 \
  --gpu-memory-utilization 0.5 --max-model-len 2048
```

In a separate terminal/pane with AgentBench's Python environment active:

```bash
cd ~/Downloads/agentbench_starter
source .venv/bin/activate
python benchmark_openai.py \
  --base-url http://127.0.0.1:8000 \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --concurrency 20 --requests 80
```

The model identifier and supported CLI options may change across vLLM-Metal versions; verify against the locally installed release.

## Next experiments

1. Add warm-up and 3–5 measured repetitions per configuration; save raw per-request measurements and a versioned JSON/CSV experiment manifest.
2. Standardize prompt texts, chat templates, `max_tokens`, decoding parameters, request counts and test duration across backends.
3. Run a controlled concurrency sweep (e.g., 1, 2, 5, 10, 20, 40) at matched workloads, alternating backend order where feasible.
4. Add streaming and measure TTFT, inter-token latency / TPOT, and total request latency separately.
5. Capture resource utilization, queue depth and server-side scheduling metrics; verify compute-device and kernel paths.
6. Add cancellation, timeouts, load shedding under sustained load, and memory-pressure / recovery tests.
7. Explore KV cache management, prefix caching and scheduler behaviors; contrast fixed-window batching with the execution mode actually provided by the installed vLLM-Metal release.
8. Later, run a separate CUDA-based vLLM benchmark on an NVIDIA GPU when studying CUDA-specific features or contributing performance changes upstream.
