"""
Tavily search wrapper, plus free no-key sources (Wikipedia, arXiv).

Tavily returns clean results and, with include_raw=True, the extracted page
content server-side (avoids scraping/403s). Wikipedia and arXiv are reliable,
never-blocked grounding sources well suited to technical/ML questions.
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
    query: str, max_results: int = 5, include_raw: bool = True, timeout: float = 30.0
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


async def wikipedia_search(query: str, timeout: float = 15.0) -> SearchResult | None:
    """Top Wikipedia page's plain-text extract. Free, no key, not blocked."""
    headers = {"User-Agent": "DeepResearchAgent/1.0 (educational project; contact: student@example.com)"}
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
        s = await client.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query,
                    "format": "json", "srlimit": 1},
        )
        s.raise_for_status()
        hits = s.json().get("query", {}).get("search", [])
        if not hits:
            return None
        title = hits[0]["title"]

        e = await client.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "prop": "extracts", "explaintext": 1,
                    "titles": title, "format": "json", "redirects": 1},
        )
        e.raise_for_status()
        pages = e.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        text = page.get("extract", "")
        if not text:
            return None

    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    return SearchResult(url=url, title=title, snippet=text[:500], raw_content=text)

async def arxiv_search(query: str, max_results: int = 2, timeout: float = 20.0) -> list[SearchResult]:
    """Top arXiv papers (title + abstract). Free, no key, not blocked. Primary
    sources for ML/retrieval topics."""
    import xml.etree.ElementTree as ET

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
    }
    # https + follow_redirects: arXiv 301-redirects http -> https.
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        r = await client.get("https://export.arxiv.org/api/query", params=params)
        r.raise_for_status()
        xml = r.text

    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml)
    out: list[SearchResult] = []
    for entry in root.findall("a:entry", ns):
        title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
        url = (entry.findtext("a:id", default="", namespaces=ns) or "").strip()
        if not summary:
            continue
        out.append(SearchResult(url=url, title=title, snippet=summary[:500],
                                raw_content=f"{title}\n\n{summary}"))
    return out