"""
Planner agent: query -> sub-questions + dependency graph.

The dependency graph is what lets the orchestrator run independent work in
parallel while still respecting "B needs A first". Output is validated into
pydantic models so a malformed LLM response fails loudly instead of silently
corrupting the run.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from src.config import settings
from src.models.llm import LLMClient
from src.observability import CostTracker

_PROMPT = (Path(__file__).parent / "prompts" / "planner.txt").read_text()


class SubQuestion(BaseModel):
    id: str
    question: str
    depends_on: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    subquestions: list[SubQuestion]


async def plan_query(query: str, llm: LLMClient, cost: CostTracker) -> Plan:
    system = _PROMPT.replace("{max_subquestions}", str(settings.max_subquestions))
    resp = await llm.generate(
        prompt=f"Research question: {query}",
        system=system,
        json_mode=True,
    )
    cost.record_llm(resp)

    # Reuse generate_json's parsing leniency by re-parsing the text.
    import json

    text = resp.text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1].lstrip("json").strip()
    data = json.loads(text)
    plan = Plan(**data)

    # Safety: cap sub-questions and drop dangling dependencies.
    plan.subquestions = plan.subquestions[: settings.max_subquestions]
    valid_ids = {sq.id for sq in plan.subquestions}
    for sq in plan.subquestions:
        sq.depends_on = [d for d in sq.depends_on if d in valid_ids and d != sq.id]
    return plan