# AgentBench — Experimental Results and Reproduction Notes

> Measurements are user-recorded local results on Apple M5 / 16 GB, MLX-LM, Qwen2.5-0.5B-Instruct-4bit. Treat them as prototype measurements, not transferable capacity guarantees. Timestamps: November 2026.

## Experiment 1: mock API concurrency (not real model execution)

| Concurrency | Requests | Throughput (req/s) | Mean latency (s) | P95 (s) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 20 | 2.77 | 0.361 | 0.362 |
| 5 | 20 | 13.37 | 0.370 | 0.392 |
| 10 | 20 | 26.56 | 0.370 | 0.377 |
| 20 | 20 | 51.09 | 0.377 | 0.381 |

Mock results characterize synthetic asynchronous request handling, **not GPU inference**.

## Experiment 2: real sequential HTTP inference

20 requests per level, 32 generated tokens per request; `model_lock` serializes model generation.

| Concurrency | Throughput (req/s) | Output tokens/s | Mean latency (ms) | P95 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8.07 | 258.37 | 124 | 137 |
| 2 | 8.49 | 271.83 | 229 | 245 |
| 5 | 8.34 | 266.77 | 534 | 617 |

### Five simultaneous requests — queue instrumentation

| Request | Lock wait (ms) | Generation (ms) | Server latency (ms) |
| --- | ---: | ---: | ---: |
| 1 | 0 | 179 | 180 |
| 2 | 178 | 120 | 298 |
| 3 | 298 | 118 | 416 |
| 4 | 416 | 120 | 537 |
| 5 | 536 | 121 | 658 |

In that observed burst, request 5 spent roughly 81% of the server-reported time waiting for the lock. Queue wait measures waiting for an application lock, **not** necessarily waiting for a GPU scheduler.

## Experiment 3: offline fixed batching

Five prompts, 32 tokens each: sequential total 1.447 s / 110.57 output tokens/s; batched total 0.421 s / 380.43 output tokens/s; speedup 3.44×. This is **direct model invocation**, not HTTP.

### Batch-size sweep (32 prompts × 32 output tokens, 3 repetitions)

Median of repeated runs from latest sweep:

| Batch size | Time (s) | Requests/s | Output tokens/s | Relative speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3.966 | 8.07 | 258.19 | 1.00× |
| 2 | 1.975 | 16.20 | 518.43 | 2.01× |
| 4 | 1.284 | 24.92 | 797.33 | 3.09× |
| 8 | 1.022 | 31.30 | 1001.55 | 3.88× |
| 16 | 0.946 | 33.81 | 1081.98 | 4.19× |
| 32 | 0.732 | 43.74 | 1399.56 | 5.42× |

Batch 32 is the *best of sizes measured for offline throughput*, not a global optimum for serving latency or memory.

## Experiment 4: online request coalescing

Server: up to 8 queued prompts collected for up to 20 ms, then one `batch_generate` call. First five-request burst: all responses reported `batch_size=5`, latency ~643–644 ms and batch wait ~21 ms; batch coalescing verified. That burst is not directly comparable with the later warmed 20-request runs.

| Concurrency | Sequential req/s | Coalesced req/s | Sequential P95 (ms) | Coalesced P95 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8.07 | 6.33 | 137 | 181 |
| 2 | 8.49 | 12.34 | 245 | 193 |
| 5 | 8.34 | 21.82 | 617 | 283 |

For concurrency 5: 2.62× observed throughput and 54% lower P95 compared with the sequential baseline. For concurrency 1: extra batch-wait overhead worsened throughput and P95. Beware run-to-run variance.

## Experiment 5: bounded queue / overload control

`MAX_BATCH_SIZE=8`, `MAX_QUEUE_SIZE=8`, fixed 32 output tokens per request. One burst per configuration:

| Concurrent clients | HTTP 200 | HTTP 503 | 503 rate | Wall time (s) |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 15 | 5 | 25% | 0.828 |
| 50 | 16 | 34 | 68% | 0.576 |
| 100 | 16 | 84 | 84% | 0.595 |

The queue bound protects the number of waiting requests; work already admitted to an executing batch is not counted in `queue_depth`. 503 responses demonstrate admission control, not guaranteed stable memory under sustained attack or bounded processing latency.

## Experiment 6: retry vs no retry — one run

50 client jobs, up to 3 retries. No retry: 15 successes, 35 final 503s, 50 attempts, 0.628 s wall, 0.623 s P95. Exponential backoff + jitter: 50 successes, 0 final 503s, 113 attempts, 63 HTTP 503 responses *across all attempts*, 3.905 s wall, 3.753 s P95. The retry policy trades time and attempts for completion.

## Experiment 7: three retry policies, 3 sequential trials each

50 jobs per policy / trial, maximum 3 retries. The **P95 includes successful and unsuccessful jobs**. Policy trials were run sequentially in rotating order; this is not a concurrent randomized A/B test.

| Trial | Policy | Final success | HTTP 503 responses | HTTP attempts | Wall (s) | P95 (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | no_retry | 16/50 | 34 | 50 | 0.60 | 0.60 |
| 1 | fixed_200ms | 32/50 | 112 | 144 | 1.05 | 1.04 |
| 1 | exponential_jitter | 50/50 | 66 | 116 | 3.61 | 3.45 |
| 2 | fixed_200ms | 32/50 | 112 | 144 | 1.10 | 1.09 |
| 2 | exponential_jitter | 50/50 | 72 | 122 | 3.62 | 3.60 |
| 2 | no_retry | 16/50 | 34 | 50 | 0.58 | 0.57 |
| 3 | exponential_jitter | 50/50 | 60 | 110 | 3.43 | 3.41 |
| 3 | no_retry | 16/50 | 34 | 50 | 0.55 | 0.55 |
| 3 | fixed_200ms | 32/50 | 112 | 144 | 1.02 | 1.01 |

### Median over three trials

| Policy | Success | Attempts | HTTP 503 | Wall (s) | P95 (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_retry | 16/50 (32%) | 50 | 34 | 0.58 | 0.57 |
| fixed_200ms | 32/50 (64%) | 144 | 112 | 1.05 | 1.04 |
| exponential_jitter | 50/50 (100%) | 116 | 66 | 3.61 | 3.45 |

Important interpretation: the fixed-delay policy caused 144 total attempts and 112 503s per 50 jobs; correlated retries may re-hit a saturated queue. In this one local workload, jittered exponential backoff completed more jobs using fewer attempts than fixed retry, at higher P95. It does **not** establish universal superiority; production policy depends on deadlines, retry budgets, admission rate, request idempotency, and client objectives.

## Reproduction and measurement quality

1. Record hardware, model revision, MLX-LM version, Python version, environment, server config, prompt/token shape, warm-up, and seeds.
2. Ensure the desired server is listening on port 8081 (`/health`); avoid conflicting older servers.
3. Test one variable at a time; save raw per-request results (status, attempts, latency, tokens, queue time), not only medians.
4. Separate cold and warm runs; randomize test order; use repeated runs and confidence intervals.
5. Report successful-request latency separately from **all-job latency** and report error rate alongside P95.
6. Confirm `Retry-After` handling, total retry budgets and cancellation safety before production use.
7. Collect model memory, process RSS/unified memory, TTFT/TPOT, per-batch size histogram, queue depth, timeouts, and sustained-load behavior in follow-up work.
