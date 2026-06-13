"""
Worker agent: research ONE sub-question.

Flow (this is the corrective-RAG loop, Milestone 3):
  1. Check memory first. Embed the sub-question, retrieve from the store.
     If we already have strong matches, we may skip the web entirely -> the key
     efficiency win over the old "always re-scrape" version.
  2. If memory is thin, search (Tavily) -> scrape -> chunk -> embed -> WRITE BACK
     to memory (so the next query benefits), then retrieve fresh.
  3. Grade the retrieved context with a lightweight LLM call. If it's not good
     enough, rewrite the query and retry (bounded by max_corrective_retries).
  4. Return the top context chunks + their sources for the synthesizer.
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
from src.tools.scrape import scrape_many
from src.tools.search import tavily_search

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
        self.cost.record_tool("tavily_search")
        results = await tavily_search(query, max_results=5)
        urls = [r.url for r in results if r.url]
        self.cost.record_tool("scrape_many")
        pages = await scrape_many(urls)

        for page in pages:
            if not page.text.strip():
                continue
            chunks = chunk_text(page.text, metadata={"url": page.url, "title": page.title})
            if not chunks:
                continue
            embeddings = await self.embedder.embed_async([c.content for c in chunks])
            await self.store.upsert_document(
                url=page.url, title=page.title, content=page.text,
                chunks=chunks, embeddings=embeddings,
            )

    async def _grade(self, sub_question: str, chunks: list[Retrieved]) -> dict:
        if not chunks:
            return {"sufficient": False, "rewritten_query": sub_question}
        context = "\n\n".join(f"[{i+1}] {c.content[:600]}" for i, c in enumerate(chunks))
        prompt = f"Sub-question: {sub_question}\n\nRetrieved context:\n{context}"
        try:
            data = await self.llm.generate_json(prompt, system=_GRADER_PROMPT)
            # generate_json doesn't return the response object, so approximate
            # cost with a tiny fixed accounting call instead:
            self.cost.llm_calls += 1
            return data
        except Exception:  # noqa: BLE001
            return {"sufficient": True}  # fail-open so a grader hiccup never blocks