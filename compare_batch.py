import time
from mlx_lm import load, generate, batch_generate

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
MAX_TOKENS = 32

questions = [
    "What is a GPU?",
    "What is LLM inference?",
    "What is a KV cache?",
    "What is model batching?",
    "What is token generation?",
]

print("Loading Qwen model...")
model, tokenizer = load(MODEL)

prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=True,
        add_generation_prompt=True
    )
    for question in questions
]

# Experiment A: one request at a time
print("\n=== Sequential Inference ===")
start = time.perf_counter()
sequential_outputs = []

for prompt in prompts:
    output = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=MAX_TOKENS,
        verbose=False
    )
    sequential_outputs.append(output)

sequential_time = time.perf_counter() - start

# Experiment B: all requests in one batch
print("\n=== Batched Inference ===")
start = time.perf_counter()

batch_result = batch_generate(
    model,
    tokenizer,
    prompts,
    max_tokens=MAX_TOKENS,
    verbose=False
)

batch_time = time.perf_counter() - start

def count_tokens(outputs):
    return sum(
        len(tokenizer.encode(t, add_special_tokens=False))
        for t in outputs
    )

seq_tokens = count_tokens(sequential_outputs)
batch_tokens = count_tokens(batch_result.texts)

print("\n========== RESULTS ==========")
print(f"Requests: {len(questions)}")
print(f"Sequential time: {sequential_time:.3f}s")
print(f"Batch time: {batch_time:.3f}s")
print(f"Sequential tokens: {seq_tokens}")
print(f"Batch tokens: {batch_tokens}")
print(f"Sequential tokens/s: {seq_tokens/sequential_time:.2f}")
print(f"Batch tokens/s: {batch_tokens/batch_time:.2f}")
print(f"Speedup (total time): {sequential_time/batch_time:.2f}x")
