from mlx_lm import load, generate
import time

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

print("Loading model...")

model, tokenizer = load(MODEL)

messages = [
    {
        "role": "user",
        "content": "What is a GPU? Answer briefly."
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

start = time.perf_counter()

response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=100
)

elapsed = time.perf_counter() - start

print("\nModel:", MODEL)
print("Response:", response)
print(f"Generation time: {elapsed:.3f}s")
