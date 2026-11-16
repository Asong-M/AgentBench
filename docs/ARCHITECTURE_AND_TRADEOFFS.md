# Architecture, Key Decisions, and Known Failure Modes

## Development trajectory

`app.py` mock API → `real_inference.py` standalone MLX inference → `real_server.py` sequential HTTP inference with model lock → `benchmark_real.py` concurrent HTTP measurements → lock-wait / generation instrumentation → `benchmark_batch_sizes.py` offline batching → `batch_server.py` online coalescing → `batch_server_backpressure.py` bounded queue + 503 → `benchmark_retry.py` and `compare_retry_policies.py` retry evaluation.

## Design decisions

### Why a model lock in the baseline?

The first server called MLX model generation through one shared model/tokenizer instance. A `threading.Lock` ensured only one generation entered that critical section at a time. This is intentionally conservative but serializes requests and makes lock wait a source of head-of-line blocking. It is **not** a measurement of hardware scheduling contention alone.

### Why fixed-size batch coalescing?

A single asynchronous worker collects requests until either (a) 8 requests are available or (b) 20 ms elapses after the first request. It calls `mlx_lm.batch_generate` in a background thread, then resolves each request's future. An asynchronous HTTP handler can receive other requests while the model generates. This is **microbatching / request coalescing**, not continuous batching that inserts and evicts sequences across token-generation steps.

### Why fix `max_tokens=32` in the bounded prototype?

A uniform output-length limit simplifies the batch interface and experiment comparability. The API currently validates this exact value; other values receive 422. Relaxing it needs deliberate testing of per-request budgets, varying prompt lengths, and correct per-request response accounting.

### Why bound the waiting queue?

An unbounded `asyncio.Queue` can grow under sustained overload. `maxsize=8` bounds the number of waiting tasks in the queue, and `put_nowait()` (or equivalent admission check) makes excess requests fail with HTTP 503 rather than waiting indefinitely. Processing batches are **outside** the waiting queue; this does not cap overall resources by itself.

### Why bounded retries with exponential backoff and jitter?

A 503 indicates temporary overload. Blind immediate retry can synchronize clients and amplify an overload. Backoff disperses retry attempts, and random jitter reduces synchronized retries. A maximum retry count limits attempts; total time budget and appropriate `Retry-After` should also be enforced. Retrying may worsen user latency and is not always desired.

## Where time is spent

A request can experience: ingress/validation + queue wait (including batching window and previously executing batches) + prompt preparation + prefill/decode generation + response serialization + HTTP/network latency. `generation_seconds` as currently returned includes execution of the batch-generation code plus worker overhead, not a GPU-only kernel measurement. HTTP client-observed latency is the preferred end-to-end signal.

## Failure mode checklist

| Failure / risk | Current status | Proposed mitigation |
| --- | --- | --- |
| Burst > queue capacity | Returns HTTP 503 (verified locally) | Retry-After, dashboards, load shedding policy |
| Burst request convergence | Coalesces up to 8 within 20 ms (verified) | Adjust window adaptively using real traffic |
| User disconnect / canceled request | Not hardened | Remove canceled items, cancel futures safely, avoid wasting GPU work |
| Waiting forever / failed worker | Not hardened | Worker supervisor, timeouts, metrics, recoverable exception paths |
| Retry storm | Three strategies evaluated | Jitter, retry budget, deadline propagation, circuit breaker |
| Crash mid-batch / model exception | Basic exception propagation | Structured errors, isolation, retries only when safe |
| Memory pressure / OOM | Not measured fully | Track unified memory, cap in-flight batches / token budgets |
| Fairness / long vs short prompts | Not characterized | FIFO fairness tests, length-aware scheduling experiments |
| Shutdown while batch running | Not validated | Drain/cancel queue safely with explicit shutdown deadline |
| Multiple workers / replicas | Not implemented | Single worker ownership, per-process isolation, routing |
| Streaming / TTFT | Not implemented | Streaming output and first-token timestamps |
| CI with Apple MLX | Local only | CPU-compatible mock tests in GitHub Actions, Apple runner if available |
| vLLM/CUDA backend | Not implemented | Separate NVIDIA GPU environment and compatible API adapter |

## Interview-ready technical invariants (public, non-personal)

- Each admitted request should receive exactly one eventual outcome (200, a bounded error, or cancellation), not an orphaned future.
- Queue depth reflects **waiting** work, not total outstanding or running work.
- HTTP 200 indicates a completed generation; HTTP 503 indicates admission rejection. The client should know whether it is safe to retry.
- Request-level status and latency distributions must be segmented by success/failure, batch size, and concurrency.
- A single performance number cannot generalize beyond the tested hardware, prompt/output distribution, and load pattern.
