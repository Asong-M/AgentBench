"""AgentBench learning server: simulated model, NO real GPU inference."""
import asyncio
import os
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="AgentBench Learning API")

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=12, ge=1, le=128)

class GenerateResponse(BaseModel):
    text: str
    generated_tokens: int
    backend: str

MOCK_WORDS = ["A", "GPU", "processes", "many", "calculations", "in", "parallel."]

@app.get("/health")
async def health():
    return {"status": "ok", "backend": os.getenv("BACKEND", "mock")}

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    backend = os.getenv("BACKEND", "mock")
    if backend == "mock":
        words = []
        for i in range(min(req.max_tokens, len(MOCK_WORDS))):
            await asyncio.sleep(0.05)  # Simulated per-token delay
            words.append(MOCK_WORDS[i])
        return GenerateResponse(text=" ".join(words), generated_tokens=len(words), backend="mock")
    if backend == "vllm":
        import httpx
        url = os.getenv("VLLM_URL", "http://127.0.0.1:8000/v1").rstrip("/")
        model = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        async with httpx.AsyncClient(timeout=180.0) as client:
            r = await client.post(f"{url}/chat/completions", json={
                "model": model, "messages": [{"role": "user", "content": req.prompt}],
                "max_tokens": req.max_tokens, "temperature": 0
            }, headers={"Authorization": "Bearer " + os.getenv("VLLM_API_KEY", "EMPTY")})
            r.raise_for_status()
            data = r.json()
        return GenerateResponse(text=data["choices"][0]["message"]["content"] or "",
                                generated_tokens=data.get("usage", {}).get("completion_tokens", 0),
                                backend="vllm")
    raise ValueError("BACKEND must be 'mock' or 'vllm'")
