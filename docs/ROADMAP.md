# AgentBench Engineering Roadmap

Statuses: **Done (local experiment)** / **Planned**. This is a learning prototype, not a production-ready inference platform.

## Completed

- [x] MLX-based Qwen local inference and FastAPI HTTP endpoint
- [x] Mock API and real-server HTTP benchmarks
- [x] Queue-wait vs generation-time instrumentation
- [x] Offline batch-size sweep 1/2/4/8/16/32
- [x] Online batched request coalescing (max 8, 20 ms window)
- [x] Bounded queue and 503 overload behavior verified at 20/50/100 concurrent clients
- [x] Retry experiment: no retry vs bounded exponential backoff + jitter
- [x] Three-policy retry comparison, 3 trials/policy
- [x] Six live-server local integration tests (observed passing)
- [x] Basic CI for mock API tests

## P0 — correctness and reproducibility

1. Standardize setup: `requirements.txt` and optional MLX dependency groups, environment version pinning, clear platform prerequisites.
2. Separate server-independent unit tests from model-dependent local integration tests; unit-test queue-full, status codes, and backoff math without downloading a model.
3. Handle cancellation, client disconnects, request deadlines, and batch-worker exceptions without leaking futures.
4. Add request IDs, structured JSON logs, and deterministic benchmark CSV/JSON output; report P50/P95/P99 separately for completed, rejected, and all jobs.
5. Add exact test provenance: commit SHA, model revision, Python/MLX versions, seeds, CPU/GPU/unified-memory sampling, warm-up policy.

## P1 — scheduling and performance

1. Measure `batch_size` histogram, queue depth, wait-to-batch, generation, and HTTP end-to-end time under Poisson/bursty arrivals.
2. Sweep `BATCH_WAIT_SECONDS` (0/5/10/20/50 ms) and `MAX_BATCH_SIZE` (1/2/4/8/16) with a latency SLO; graph throughput-vs-P95 Pareto frontier.
3. Measure longer/mixed prompt lengths, variable output lengths, token fairness, and memory consumption. Avoid assuming offline batch 32 is optimal online.
4. Add TTFT and TPOT (requires streaming or token-step instrumentation; not equivalent to current total latency).
5. Explore cancellation-aware admission control, adaptive batching, separate pending/active token budgets, fair scheduling.

## P2 — operations and resilience

1. Deadline-aware bounded retry budget with `Retry-After` (including HTTP-date), full jitter, idempotency, and circuit breaker.
2. Sustained-load stress / soak tests (5–30 min), throughput/error/latency monitoring and memory leakage checks.
3. Graceful shutdown: stop admissions, finish or cancel accepted requests with defined semantics.
4. Containerization for backend-independent components; Apple MLX compatibility varies by runtime. Never claim CUDA support from Mac tests.
5. Add privacy/security protections for prompts, request size limits, rate limits, abuse protection.

## P3 — vLLM / SGLang learning path

1. Learn prefill vs decode, KV cache layout, PagedAttention, batching scheduler and memory tradeoffs.
2. Provision a **separate** NVIDIA CUDA GPU environment to run an official vLLM server; do not use the current MLX code as if it were vLLM.
3. Build a common OpenAI-compatible HTTP benchmark adapter; measure at comparable model, prompt/output distribution and latency objectives where possible.
4. Investigate a focused upstream issue and propose a small, reproducible vLLM PR (tests/docs/bug fix); only claim upstream contribution if actually accepted.

## Recommended acceptance gates

- Server never leaves a request future unresolved under cancellation/errors.
- Queue limit and max in-flight tokens remain bounded in stress tests.
- Every benchmark records success rate, retry amplification, P95 by outcome, and hardware/software versions.
- CI passes on a clean clone without requiring a locally running MLX model.
