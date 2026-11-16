"""OpenAI-style chat completions adapter around AgentBench's existing MLX app.

Run: python -m uvicorn openai_adapter:app --host 127.0.0.1 --port 8081
The original /generate and /health routes remain available.

This is a minimal compatibility layer, not full OpenAI API semantics:
- only non-streaming text messages
- fixed 32-token budget inherited from the underlying MLX server
- roles converted to a simple text prompt (not a model chat template)
- no tokenized prompt usage accounting
"""
from __future__ import annotations

import time
import uuid
from typing import Literal

import httpx
from fastapi import HTTPException, Response
from pydantic import BaseModel, Field

from batch_server_backpressure import app


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = 32
    stream: bool = False


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, response: Response):
    if req.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported by this adapter")
    if req.max_tokens != 32:
        raise HTTPException(status_code=422, detail="This MLX backend currently requires max_tokens=32")

    # Experimental role-to-text conversion. This is NOT Qwen's real chat template.
    prompt = "\n".join(f"{message.role.capitalize()}: {message.content}" for message in req.messages)
    prompt += "\nAssistant:"

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://agentbench.local") as client:
        upstream = await client.post("/generate", json={"prompt": prompt, "max_tokens": req.max_tokens})

    if upstream.status_code != 200:
        retry_after = upstream.headers.get("retry-after")
        if retry_after is not None:
            response.headers["Retry-After"] = retry_after
        # Preserve admission control failures (503) and validation failures (422).
        try:
            detail = upstream.json()
        except ValueError:
            detail = upstream.text
        raise HTTPException(status_code=upstream.status_code, detail=detail, headers={"Retry-After": retry_after} if retry_after else None)

    data = upstream.json()
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": data.get("model", req.model),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": data["text"]},
            "finish_reason": None,  # Underlying server does not expose finish reason.
        }],
        "usage": {"completion_tokens": data.get("generated_tokens")},
        "agentbench_metrics": {
            key: data[key]
            for key in ("backend", "batch_size", "queue_wait_seconds", "generation_seconds", "latency_seconds")
            if key in data
        },
    }
