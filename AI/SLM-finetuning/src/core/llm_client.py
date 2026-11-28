import asyncio
import json
import re
from typing import Any

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import LLMConfig

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(raw_text: str) -> Any:
    """Safely parses JSON even if the model wraps it in markdown fences or extra text."""
    text = raw_text.strip()
    match = _JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for open_ch, close_ch in (("[", "]"), ("{", "}")):
            start = text.find(open_ch)
            end = text.rfind(close_ch)
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
        raise


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

    @retry(wait=wait_exponential(min=2, max=30), stop=stop_after_attempt(4))
    async def _call_api(self, system_prompt: str, user_prompt: str, temperature: float) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": 4096,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if "gpt-oss" in self.model:
            kwargs["reasoning_effort"] = "low"

        response = await self._client.chat.completions.create(**kwargs)
        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise ValueError("Empty response from model")
        return extract_json(content)

    async def generate_json(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.2
    ) -> Any:
        async with self._semaphore:
            result = await self._call_api(system_prompt, user_prompt, temperature)
            if self._min_delay > 0:
                await asyncio.sleep(self._min_delay)
            return result

    async def close(self) -> None:
        await self._client.close()