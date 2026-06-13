"""
Synthesizer agent: worker contexts -> a single cited report.

Grounding rule (the thing that crushes hallucination, and the thing your
faithfulness metric measures): the model may state ONLY facts present in the
provided context, and must cite each one. We assemble a numbered source list so
citations are checkable against must_cite_sources in the eval set.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.models.llm import LLMClient
from src.observability import CostTracker, trace

_PROMPT = (Path(__file__).parent / "prompts" / "synthesizer.txt").read_text()


@dataclass
class Report:
    text: str
    sources: list[str]  # ordered; index i -> citation marker [i+1]


def _build_context(worker_results) -> tuple[str, list[str]]:
    """Flatten all worker chunks into a numbered context block + source list."""
    seen: dict[str, int] = {}
    sources: list[str] = []
    blocks: list[str] = []
    for wr in worker_results:
        for c in wr.chunks:
            url = c.source_url or "unknown"
            if url not in seen:
                seen[url] = len(sources) + 1
                sources.append(url)
            n = seen[url]
            blocks.append(f"[{n}] (from {c.source_title or url})\n{c.content}")
    return "\n\n".join(blocks), sources


@trace("synthesizer")
async def synthesize(
    query: str, worker_results, llm: LLMClient, cost: CostTracker
) -> Report:
    context, sources = _build_context(worker_results)
    if not context.strip():
        return Report(
            text="The retrieved sources do not contain enough information to answer this question.",
            sources=[],
        )
    source_map = "\n".join(f"[{i+1}] {u}" for i, u in enumerate(sources))
    prompt = (
        f"Original question: {query}\n\n"
        f"Numbered sources:\n{source_map}\n\n"
        f"Context passages:\n{context}\n\n"
        "Write the cited report now."
    )
    resp = await llm.generate(prompt, system=_PROMPT, temperature=0.1)
    cost.record_llm(resp)
    return Report(text=resp.text, sources=sources)