"""Send one inference request to AgentBench."""
import asyncio
import httpx

async def main():
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post("http://127.0.0.1:8080/generate", json={
            "prompt": "What is a GPU?", "max_tokens": 12
        })
        response.raise_for_status()
        print(response.json())

if __name__ == "__main__":
    asyncio.run(main())
