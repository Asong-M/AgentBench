"""AgentBench: simple online fixed-size MLX batching server.

Start from the AgentBench project directory:
    python -m uvicorn batch_server:app --host 127.0.0.1 --port 8081

This learning prototype supports a fixed 32-token generation budget per request.
It implements request coalescing (not production continuous batching).
"""

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from mlx_lm import batch_generate, load
from pydantic import BaseModel, Field

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
MAX_BATCH_SIZE = 8
BATCH_WAIT_SECONDS = 0.020
MAX_TOKENS = 32

model = None
tokenizer = None
request_queue = None


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    # Keep length fixed so each request receives the same generation budget.
    max_tokens: int = Field(default=MAX_TOKENS, ge=MAX_TOKENS, le=MAX_TOKENS)


def run_batch(batch):
    """Run a batch on a background thread so HTTP can remain responsive."""
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
        )
        for item in batch
    ]
    return batch_generate(
        model,
        tokenizer,
        prompts,
        max_tokens=MAX_TOKENS,
        verbose=False,
        return_token_ids=True,
    )


async def batch_worker():
    while True:
        first = await request_queue.get()
        batch = [first]
        deadline = time.perf_counter() + BATCH_WAIT_SECONDS

        while len(batch) < MAX_BATCH_SIZE:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(request_queue.get(), timeout=remaining)
                batch.append(item)
            except asyncio.TimeoutError:
                break

        generation_start = time.perf_counter()
        try:
            result = await asyncio.to_thread(run_batch, batch)
            generation_seconds = time.perf_counter() - generation_start

            for index, item in enumerate(batch):
                if not item["future"].done():
                    item["future"].set_result(
                        {
                            "text": result.texts[index],
                            "backend": "mlx-batch",
                            "model": MODEL,
                            "generated_tokens": len(result.token_ids[index]),
                            "batch_size": len(batch),
                            "queue_wait_seconds": round(
                                generation_start - item["arrival"], 4
                            ),
                            "generation_seconds": round(generation_seconds, 4),
                        }
                    )
        except Exception as exc:
            for item in batch:
                if not item["future"].done():
                    item["future"].set_exception(exc)
        finally:
            for _ in batch:
                request_queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, request_queue
    print("Loading Qwen for Dynamic Batching...")
    model, tokenizer = load(MODEL)
    request_queue = asyncio.Queue()
    worker = asyncio.create_task(batch_worker())
    print("Batch server ready!")
    try:
        yield
    finally:
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass


app = FastAPI(title="AgentBench Dynamic Batching", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "backend": "mlx-batch",
        "max_batch_size": MAX_BATCH_SIZE,
        "max_tokens": MAX_TOKENS,
    }


@app.post("/generate")
async def generate_text(req: GenerateRequest):
    arrival = time.perf_counter()
    future = asyncio.get_running_loop().create_future()
    await request_queue.put(
        {
            "prompt": req.prompt,
            "arrival": arrival,
            "future": future,
        }
    )
    result = await future
    result["latency_seconds"] = round(time.perf_counter() - arrival, 4)
    return result
