"""
Worker agent: research ONE sub-question.

Flow (corrective-RAG loop, Milestone 3):
  1. Memory-first: retrieve from the store; if strong matches exist, skip the web.
  2. Else fetch -> chunk -> embed -> write back to memory, then retrieve.
     Fetch prefers Tavily's server-side content (raw_content), adds Wikipedia +
     arXiv as reliable sources, and only scrapes URLs Tavily didn't extract.
  3. Grade the context; if weak, rewrite the query and retry (bounded).
  4. Return top chunks + sources for the synthesizer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.config import settings
from src.memory.chunker import chunk_text
from src.memory.retrieval import Retriever
from src.memory.store import MemoryStore, Retrieved
from src.models.embeddings import EmbeddingClient
from src.models.llm import LLMClient
from src.observability import CostTracker, trace
from src.tools.scrape import scrape_url
from src.tools.search import tavily_search, wikipedia_search, arxiv_search

_GRADER_PROMPT = (Path(__file__).parent / "prompts" / "worker_grader.txt").read_text()


@dataclass
class WorkerResult:
    sub_question: str
    chunks: list[Retrieved]
    used_memory_only: bool = False
    retries: int = 0
    notes: list[str] = field(default_factory=list)


class Worker:
    def __init__(
        self,
        store: MemoryStore,
        retriever: Retriever,
        embedder: EmbeddingClient,
        llm: LLMClient,
        cost: CostTracker,
        retrieval_strategy: str = "rerank",
    ):
        self.store = store
        self.retriever = retriever
        self.embedder = embedder
        self.llm = llm
        self.cost = cost
        self.strategy = retrieval_strategy

    @trace("worker")
    async def run(self, sub_question: str) -> WorkerResult:
        result = WorkerResult(sub_question=sub_question, chunks=[])
        query = sub_question

        # 1) Memory-first check.
        memory_hits = await self.retriever.retrieve(query, strategy="dense_only")
        strong = [c for c in memory_hits.chunks if c.score >= settings.memory_hit_threshold]
        if len(strong) >= settings.retrieval_k:
            result.used_memory_only = True
            result.notes.append("answered from memory (no web fetch)")
            full = await self.retriever.retrieve(query, strategy=self.strategy)
            result.chunks = full.chunks
            return result

        # 2) Fetch what's missing, then 3) grade + corrective retry.
        for attempt in range(settings.max_corrective_retries + 1):
            await self._fetch_and_store(query)
            retrieved = await self.retriever.retrieve(query, strategy=self.strategy)
            result.chunks = retrieved.chunks

            grade = await self._grade(sub_question, retrieved.chunks)
            if grade.get("sufficient") or attempt == settings.max_corrective_retries:
                if not grade.get("sufficient"):
                    result.notes.append("retries exhausted; using best available context")
                break

            # Corrective step: rewrite and retry.
            result.retries += 1
            query = grade.get("rewritten_query") or query
            result.notes.append(f"corrective retry -> '{query[:60]}'")

        return result

    async def _fetch_and_store(self, query: str):
        """Gather page texts, preferring sources that don't need scraping."""
        self.cost.record_tool("tavily_search")
        results = await tavily_search(query, max_results=5, include_raw=True)

        docs: list[tuple[str, str, str]] = []          # (url, title, text)
        to_scrape: list[tuple[str, str]] = []          # (url, title) needing a scrape

        for r in results:
            if not r.url:
                continue
            if r.raw_content and r.raw_content.strip():
                docs.append((r.url, r.title, r.raw_content))   # Tavily already has it
            else:
                to_scrape.append((r.url, r.title))             # fall back to scraping

        # Scrape only the ones Tavily didn't return content for.
        if to_scrape:
            self.cost.record_tool("scrape")
            for url, title in to_scrape:
                page = await scrape_url(url)
                if page and page.text.strip():
                    docs.append((page.url, page.title or title, page.text))

        # Wikipedia: reliable, never blocked.
        try:
            self.cost.record_tool("wikipedia")
            wiki = await wikipedia_search(query)
            if wiki and wiki.raw_content:
                docs.append((wiki.url, wiki.title, wiki.raw_content))
        except Exception:  # noqa: BLE001 - bonus source; never block on it
            pass

        # arXiv: primary sources for ML/retrieval topics.
        try:
            self.cost.record_tool("arxiv")
            for paper in await arxiv_search(query, max_results=2):
                if paper.raw_content:
                    docs.append((paper.url, paper.title, paper.raw_content))
        except Exception:  # noqa: BLE001 - bonus source; never block on it
            pass

        # Chunk + embed + write everything back to memory.
        for url, title, text in docs:
            text = text.replace("\x00", "")[:5000]
            if not text.strip():
                continue
            chunks = chunk_text(text, metadata={"url": url, "title": title})
            if not chunks:
                continue
            embeddings = await self.embedder.embed_async([c.content for c in chunks])
            await self.store.upsert_document(
                url=url, title=title, content=text,
                chunks=chunks, embeddings=embeddings,
            )

    async def _grade(self, sub_question: str, chunks: list[Retrieved]) -> dict:
        if not chunks:
            return {"sufficient": False, "rewritten_query": sub_question}
        context = "\n\n".join(f"[{i+1}] {c.content[:600]}" for i, c in enumerate(chunks))
        prompt = f"Sub-question: {sub_question}\n\nRetrieved context:\n{context}"
        try:
            data = await self.llm.generate_json(prompt, system=_GRADER_PROMPT)
            self.cost.llm_calls += 1
            return data
        except Exception:  # noqa: BLE001
            return {"sufficient": True}  # fail-open so a grader hiccup never blocks