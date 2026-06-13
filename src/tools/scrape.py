"""
Scraper: fetch a URL and return clean text.

httpx for async fetching, BeautifulSoup for extraction. We strip script/style/
nav/footer noise and collapse whitespace so the chunker gets readable prose.
Retries with exponential backoff handle flaky pages and rate limits; failures
return None so one bad URL never kills a whole research run (graceful
degradation, Milestone 4).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; DeepResearchAgent/1.0; +https://example.com/bot)"
    )
}
_NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "noscript", "form"]


@dataclass
class ScrapedPage:
    url: str
    title: str
    text: str


async def scrape_url(
    url: str, timeout: float = 20.0, retries: int = 3
) -> ScrapedPage | None:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=True, headers=_HEADERS
            ) as client:
                r = await client.get(url)
                r.raise_for_status()
                return _extract(url, r.text)
        except Exception as exc:  # noqa: BLE001 - we want to retry on anything transient
            last_exc = exc
            await asyncio.sleep(0.5 * (2 ** attempt))  # 0.5s, 1s, 2s backoff
    print(f"[scrape] giving up on {url}: {last_exc}")
    return None


def _extract(url: str, html: str) -> ScrapedPage:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    title = (soup.title.string or "").strip() if soup.title else ""
    text = soup.get_text(separator="\n")
    # Collapse blank lines / runaway whitespace.
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    return ScrapedPage(url=url, title=title, text=text)


async def scrape_many(urls: list[str]) -> list[ScrapedPage]:
    results = await asyncio.gather(*(scrape_url(u) for u in urls))
    return [p for p in results if p is not None]