"""
Tavily search wrapper.

Tavily returns clean, LLM-friendly results (url, title, a content snippet, and
optionally raw page content). We use it to discover sources to scrape. Keeping
it behind a small dataclass interface means you could swap in another search
provider without touching the agents.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx

from src.config import settings


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    raw_content: str | None = None


async def tavily_search(
    query: str, max_results: int = 5, include_raw: bool = False, timeout: float = 30.0
) -> list[SearchResult]:
    if not settings.tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY is not set (see .env.example).")
    body = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced",
        "include_raw_content": include_raw,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post("https://api.tavily.com/search", json=body)
        r.raise_for_status()
        data = r.json()

    return [
        SearchResult(
            url=item.get("url", ""),
            title=item.get("title", ""),
            snippet=item.get("content", ""),
            raw_content=item.get("raw_content"),
        )
        for item in data.get("results", [])
    ]