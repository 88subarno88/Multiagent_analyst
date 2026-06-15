"""
Scraper: fetch a URL and return clean text.

httpx for async fetching, BeautifulSoup for extraction. Strips script/style/nav
noise and collapses whitespace. Failures return None so one bad URL never kills
a research run (graceful degradation). Browser-like headers reduce bot blocks;
permanent blocks (401/403) are not retried.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}
_NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "noscript", "form"]

# Status codes that mean "blocked / won't work" — don't waste retries on these.
_NO_RETRY_STATUS = {401, 403, 404, 410, 451}


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
                # Permanent blocks: give up immediately, no retries.
                if r.status_code in _NO_RETRY_STATUS:
                    print(f"[scrape] blocked ({r.status_code}) {url}")
                    return None
                r.raise_for_status()
                return _extract(url, r.text)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in _NO_RETRY_STATUS:
                print(f"[scrape] blocked ({exc.response.status_code}) {url}")
                return None
            last_exc = exc
            await asyncio.sleep(0.5 * (2 ** attempt))
        except Exception as exc:  # transient (timeout, connection reset, etc.)
            last_exc = exc
            await asyncio.sleep(0.5 * (2 ** attempt))
    print(f"[scrape] giving up on {url}: {last_exc}")
    return None


def _extract(url: str, html: str) -> ScrapedPage:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    title = (soup.title.string or "").strip() if soup.title else ""
    text = soup.get_text(separator="\n")
    # Postgres rejects NUL bytes (\x00) in text columns; strip them before storing.
    text = text.replace("\x00", "")
    # Collapse blank lines / runaway whitespace.
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    return ScrapedPage(url=url, title=title, text=text)


async def scrape_many(urls: list[str]) -> list[ScrapedPage]:
    results = await asyncio.gather(*(scrape_url(u) for u in urls))
    return [p for p in results if p is not None]
