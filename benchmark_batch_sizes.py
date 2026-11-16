import statistics
import time

from mlx_lm import load, batch_generate

# Experiment configuration
MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
TOTAL_REQUESTS = 32
MAX_TOKENS = 32
REPEATS = 3

# 32 fixed prompts shared by every experiment
topics = [
    "GPU inference",
    "KV cache",
    "model batching",
    "token generation",
    "LLM serving",
    "GPU memory",
    "transformers",
    "inference latency",
]

questions = [
    f"Explain {topics[i % len(topics)]} "
    f"in one sentence for a beginner. Example {i + 1}."
    for i in range(TOTAL_REQUESTS)
]

print("Loading model...")
model, tokenizer = load(MODEL)
print("Model ready!")

# Prepare the same input prompts for all tests
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": q}],
        tokenize=True,
        add_generation_prompt=True
    )
    for q in questions
]


def run_experiment(batch_size):
    """Process all 32 requests in fixed-size batches."""

    start = time.perf_counter()
    total_output_tokens = 0
    completed = 0

    for i in range(0, TOTAL_REQUESTS, batch_size):
        batch_prompts = prompts[i:i + batch_size]

        result = batch_generate(
            model,
            tokenizer,
            batch_prompts,
            max_tokens=MAX_TOKENS,
            verbose=False,
            return_token_ids=True
        )

        completed += len(result.texts)

        # Count actual generated token IDs
        total_output_tokens += sum(
            len(ids) for ids in result.token_ids
        )

    elapsed = time.perf_counter() - start

    return {
        "batch_size": batch_size,
        "requests": completed,
        "time": elapsed,
        "tokens": total_output_tokens,
        "req_per_sec": completed / elapsed,
        "tokens_per_sec": total_output_tokens / elapsed
    }


# Warm-up: avoids including first-run overhead
print("\nWarming up model...")
batch_generate(
    model,
    tokenizer,
    prompts[:1],
    max_tokens=MAX_TOKENS,
    verbose=False
)

all_results = {}

for batch_size in BATCH_SIZES:
    print(f"\nTesting Batch Size = {batch_size}")

    runs = []

    for repeat in range(REPEATS):
        result = run_experiment(batch_size)
        runs.append(result)

        print(
            f"  Run {repeat + 1}: "
            f"{result['time']:.3f}s, "
            f"{result['tokens_per_sec']:.2f} tokens/s"
        )

    all_results[batch_size] = runs


print("\n" + "=" * 70)
print("BATCH SIZE SCALING RESULTS")
print("=" * 70)

print(
    f"{'Batch':<8}"
    f"{'Time(s)':<12}"
    f"{'Req/s':<12}"
    f"{'Tokens/s':<14}"
    f"{'Speedup':<10}"
)

baseline_time = statistics.median(
    r["time"] for r in all_results[1]
)

for batch_size in BATCH_SIZES:
    runs = all_results[batch_size]

    median_time = statistics.median(
        r["time"] for r in runs
    )

    median_req_s = statistics.median(
        r["req_per_sec"] for r in runs
    )

    median_tok_s = statistics.median(
        r["tokens_per_sec"] for r in runs
    )

    speedup = baseline_time / median_time

    print(
        f"{batch_size:<8}"
        f"{median_time:<12.3f}"
        f"{median_req_s:<12.2f}"
        f"{median_tok_s:<14.2f}"
        f"{speedup:<10.2f}x"
    )

print("\nAll experiments completed.")
