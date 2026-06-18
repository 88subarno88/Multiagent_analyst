"""
Scraper: fetch a URL and return clean text.

httpx + BeautifulSoup. Browser-like headers reduce bot blocks; when a fetch is
blocked (403 etc.) or fails, we retry through Jina Reader (r.jina.ai), which
fetches server-side and returns clean text, dodging most blocks. Failures return
None so one bad URL never kills a research run.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
}
_NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "noscript", "form"]
_NO_RETRY_STATUS = {401, 403, 404, 410, 451}


@dataclass
class ScrapedPage:
    url: str
    title: str
    text: str


async def _jina_fetch(url: str, timeout: float = 25.0) -> ScrapedPage | None:
    """Rescue fetch via Jina Reader: clean text fetched server-side."""
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            r = await client.get(f"https://r.jina.ai/{url}",
                                 headers={"User-Agent": _HEADERS["User-Agent"]})
            r.raise_for_status()
            text = r.text.replace("\x00", "")
            lines = [ln.strip() for ln in text.splitlines()]
            text = "\n".join(ln for ln in lines if ln)
            if not text.strip():
                return None
            return ScrapedPage(url=url, title="", text=text)
    except Exception as exc:  # noqa: BLE001
        print(f"[scrape] jina fallback failed {url}: {exc}")
        return None


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
                if r.status_code in _NO_RETRY_STATUS:
                    print(f"[scrape] blocked ({r.status_code}) {url} -> trying jina")
                    return await _jina_fetch(url)
                r.raise_for_status()
                return _extract(url, r.text)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in _NO_RETRY_STATUS:
                print(f"[scrape] blocked ({exc.response.status_code}) {url} -> trying jina")
                return await _jina_fetch(url)
            last_exc = exc
            await asyncio.sleep(0.5 * (2 ** attempt))
        except Exception as exc:  # transient (timeout, connection reset, etc.)
            last_exc = exc
            await asyncio.sleep(0.5 * (2 ** attempt))
    print(f"[scrape] giving up on direct fetch {url}: {last_exc} -> trying jina")
    return await _jina_fetch(url)


def _extract(url: str, html: str) -> ScrapedPage:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    title = (soup.title.string or "").strip() if soup.title else ""
    text = soup.get_text(separator="\n")
    text = text.replace("\x00", "")
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    return ScrapedPage(url=url, title=title, text=text)


async def scrape_many(urls: list[str]) -> list[ScrapedPage]:
    results = await asyncio.gather(*(scrape_url(u) for u in urls))
    return [p for p in results if p is not None]