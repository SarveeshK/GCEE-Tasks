import asyncio
import json
import re
from typing import Any

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import LLMConfig

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(raw_text: str) -> Any:
    """Safely parses JSON even if the model wraps it in markdown code fences."""
    text = raw_text.strip()
    match = _JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


class AsyncLLMClient:
    """Rate-limited, retry-enabled async wrapper for OpenRouter / OpenAI-compatible APIs."""

    def __init__(self, cfg: LLMConfig, max_concurrency: int = 2, min_delay: float = 1.5) -> None:
        self.model = cfg.model
        self._min_delay = min_delay
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._client = AsyncOpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            default_headers={"X-Title": "CCAC-OGMP-SLM-Pipeline"},
        )

    @retry(wait=wait_exponential(min=4, max=60), stop=stop_after_attempt(6))
    async def _call_api(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def generate_json(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.2
    ) -> Any:
        async with self._semaphore:
            raw = await self._call_api(system_prompt, user_prompt, temperature)
            if self._min_delay > 0:
                await asyncio.sleep(self._min_delay)
            return extract_json(raw)

    async def close(self) -> None:
        await self._client.close()
