import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field
from mlx_lm import load, generate

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

model = None
tokenizer = None
model_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    print("Loading real LLM...")
    model, tokenizer = load(MODEL)
    print("Model ready!")
    yield


app = FastAPI(
    title="AgentBench Real Inference",
    lifespan=lifespan
)


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=32, ge=1, le=256)


@app.get("/health")
def health():
    return {"status": "ok", "backend": "mlx"}


@app.post("/generate")
def generate_text(req: GenerateRequest):
    start = time.perf_counter()

    messages = [
        {"role": "user", "content": req.prompt}
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    lock_start = time.perf_counter()

    with model_lock:
        queue_wait = time.perf_counter() - lock_start
        generation_start = time.perf_counter()

        output = generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=req.max_tokens,
            verbose=False
        )

        generation_time = time.perf_counter() - generation_start

    latency = time.perf_counter() - start

    # 重新编码生成文本，估算输出 Token 数
    generated_tokens = len(
        tokenizer.encode(
            output,
            add_special_tokens=False
        )
    )

    return {
        "text": output,
        "backend": "mlx",
        "model": MODEL,
        "generated_tokens": generated_tokens,
        "latency_seconds": round(latency, 4),
        "queue_wait_seconds": round(queue_wait, 4),
        "generation_seconds": round(generation_time, 4)
    }
