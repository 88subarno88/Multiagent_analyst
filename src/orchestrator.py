"""
Orchestrator — the brain.

Runs the planner, then executes worker agents respecting the dependency graph:
independent sub-questions run concurrently; dependent ones wait for their
prerequisites. Finally the synthesizer produces a grounded, cited report. The
whole run shares one CostTracker so you get per-query token/$ totals.

The `retrieval_strategy` argument is what makes this benchmarkable: run the
exact same pipeline with "dense_only", "hybrid", "rerank" and compare scores.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from src.agents.planner import Plan, plan_query
from src.agents.synthesizer import Report, synthesize
from src.agents.worker import Worker, WorkerResult
from src.config import settings
from src.memory.retrieval import Retriever
from src.memory.store import MemoryStore, store as default_store
from src.models.embeddings import EmbeddingClient
from src.models.llm import LLMClient
from src.observability import BudgetExceeded, CostTracker, trace


@dataclass
class RunResult:
    query: str
    plan: Plan
    worker_results: list[WorkerResult]
    report: Report
    cost: dict
    strategy: str
    error: str | None = None


class Orchestrator:
    def __init__(
        self,
        store: MemoryStore | None = None,
        provider: str | None = None,
        retrieval_strategy: str = "rerank",
    ):
        self.store = store or default_store
        self.embedder = EmbeddingClient()
        self.retriever = Retriever(self.store, self.embedder)
        self.llm = LLMClient(provider=provider)
        self.strategy = retrieval_strategy

    async def setup(self):
        await self.store.init_schema()

    @trace("orchestrator.run")
    async def run(self, query: str) -> RunResult:
        cost = CostTracker()
        try:
            plan = await plan_query(query, self.llm, cost)
            worker_results = await self._run_workers(plan, cost)
            report = await synthesize(query, worker_results, self.llm, cost)
            return RunResult(query, plan, worker_results, report,
                             cost.summary(), self.strategy)
        except BudgetExceeded as exc:
            return RunResult(
                query, Plan(subquestions=[]), [],
                Report(text=f"Stopped: {exc}", sources=[]),
                cost.summary(), self.strategy, error=str(exc),
            )

    async def _run_workers(self, plan: Plan, cost: CostTracker) -> list[WorkerResult]:
        worker = Worker(
            self.store, self.retriever, self.embedder, self.llm, cost, self.strategy
        )
        results: dict[str, WorkerResult] = {}
        by_id = {sq.id: sq for sq in plan.subquestions}
        pending = set(by_id)

        # Topological waves: each pass runs every sub-question whose deps are done.
        while pending:
            ready = [
                sid for sid in pending
                if all(dep in results for dep in by_id[sid].depends_on)
            ]
            if not ready:  # dependency cycle safety valve -> run the rest flat
                ready = list(pending)
            coros = [worker.run(by_id[sid].question) for sid in ready]
            done = await asyncio.gather(*coros)
            for sid, res in zip(ready, done):
                results[sid] = res
                pending.discard(sid)

        return [results[sq.id] for sq in plan.subquestions]


async def _demo():
    """`python -m src.orchestrator "your question"` for a quick end-to-end test."""
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "What is Reciprocal Rank Fusion and why is it used in hybrid search?"
    orch = Orchestrator(retrieval_strategy="rerank")
    await orch.setup()
    result = await orch.run(query)
    print("\n=== PLAN ===")
    for sq in result.plan.subquestions:
        print(f"  {sq.id}: {sq.question}  (deps={sq.depends_on})")
    print("\n=== REPORT ===\n")
    print(result.report.text)
    print("\n=== COST ===")
    print(result.cost)
    await orch.store.close()


if __name__ == "__main__":
    asyncio.run(_demo())