"""
Provider-agnostic LLM client.

Uses raw httpx against each provider's REST API rather than a heavy SDK, so the
dependency surface stays small and the request shape is visible (good for
interviews: you can explain exactly what a 'tool call' or 'json mode' is on the
wire). Returns a uniform LLMResponse with token counts so the cost tracker works
the same regardless of provider.
"""
from __future__ import annotations

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


class LLMClient:
    def __init__(self, provider: str | None = None):
        self.provider = provider or settings.primary_provider

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
        timeout: float = 90.0,
    ) -> LLMResponse:
        if self.provider == "gemini":
            return await self._gemini(prompt, system, json_mode, temperature, timeout)
        if self.provider == "ollama":
            return await self._ollama(prompt, system, json_mode, temperature, timeout)
        raise ValueError(f"Unknown provider: {self.provider}")

    async def generate_json(self, prompt: str, system: str | None = None, **kw) -> dict:
        """Convenience wrapper that parses JSON-mode output defensively."""
        resp = await self.generate(prompt, system=system, json_mode=True, **kw)
        text = resp.text.strip()
        # Strip accidental markdown fences if the model adds them.
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Last-ditch: grab the outermost {...} or [...].
            for open_c, close_c in (("{", "}"), ("[", "]")):
                i, j = text.find(open_c), text.rfind(close_c)
                if i != -1 and j != -1:
                    try:
                        return json.loads(text[i : j + 1])
                    except json.JSONDecodeError:
                        continue
            raise

    # ---- Gemini ----
    async def _gemini(self, prompt, system, json_mode, temperature, timeout) -> LLMResponse:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set (see .env.example).")
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

        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                url,
                headers={"x-goog-api-key": settings.gemini_api_key},
                json=body,
            )
            r.raise_for_status()
            data = r.json()

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

    # ---- Ollama (local open-weight) ----
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