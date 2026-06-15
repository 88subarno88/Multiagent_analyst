"""
Provider-agnostic LLM client.

Raw httpx against each provider's REST API. Gemini rotates across multiple API
keys on 429 and backs off when all are limited. Ollama runs a local model.
Returns a uniform LLMResponse with token counts.
"""
from __future__ import annotations

import asyncio
import itertools
import json
from dataclasses import dataclass

import httpx

from src.config import settings


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    provider: str = ""


_key_cycle = itertools.cycle(settings.gemini_key_list) if settings.gemini_key_list else None


async def _gemini_post(url: str, body: dict, timeout: float, max_rounds: int = 5) -> dict:
    """POST to Gemini, rotating keys on 429 and backing off when all are limited."""
    keys = settings.gemini_key_list
    if not keys:
        raise RuntimeError("No Gemini API keys configured (set GEMINI_API_KEY or GEMINI_API_KEYS).")

    async with httpx.AsyncClient(timeout=timeout) as client:
        for round_ in range(max_rounds):
            for _ in range(len(keys)):
                key = next(_key_cycle)
                r = await client.post(url, headers={"x-goog-api-key": key}, json=body)
                if r.status_code == 429:
                    continue
                r.raise_for_status()
                return r.json()
            await asyncio.sleep(2 ** round_)
        r = await client.post(url, headers={"x-goog-api-key": next(_key_cycle)}, json=body)
        r.raise_for_status()
        return r.json()


class LLMClient:
    def __init__(self, provider: str | None = None):
        self.provider = provider or settings.primary_provider

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
        timeout: float = 600.0,
    ) -> LLMResponse:
        if self.provider == "gemini":
            return await self._gemini(prompt, system, json_mode, temperature, timeout)
        if self.provider == "ollama":
            return await self._ollama(prompt, system, json_mode, temperature, timeout)
        raise ValueError(f"Unknown provider: {self.provider}")

    async def generate_json(self, prompt: str, system: str | None = None, **kw) -> dict:
        """Call generate in JSON mode and parse defensively."""
        resp = await self.generate(prompt, system=system, json_mode=True, **kw)
        text = resp.text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            for open_c, close_c in (("{", "}"), ("[", "]")):
                i, j = text.find(open_c), text.rfind(close_c)
                if i != -1 and j != -1:
                    try:
                        return json.loads(text[i : j + 1])
                    except json.JSONDecodeError:
                        continue
            raise

    async def _gemini(self, prompt, system, json_mode, temperature, timeout) -> LLMResponse:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{settings.gemini_model}:generateContent"
        )
        body: dict = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"

        data = await _gemini_post(url, body, timeout)

        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            text = ""
        usage = data.get("usageMetadata", {})
        return LLMResponse(
            text=text,
            input_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0),
            model=settings.gemini_model,
            provider="gemini",
        )

    async def _ollama(self, prompt, system, json_mode, temperature, timeout) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": settings.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            body["format"] = "json"
            messages[-1]["content"] += "\n\nRespond with valid JSON only."

        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{settings.ollama_base_url}/api/chat", json=body)
            r.raise_for_status()
            data = r.json()

        return LLMResponse(
            text=data.get("message", {}).get("content", ""),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            model=settings.ollama_model,
            provider="ollama",
        )
