from __future__ import annotations
import argparse
import asyncio
import json
import time
from pathlib import Path
from statistics import mean
from evals.metrics import answer_relevance, citation_accuracy, faithfulness
from src.config import settings
from src.models.llm import LLMClient
from src.orchestrator import Orchestrator

_DATASET = Path(__file__).parent / "dataset.jsonl"
_RESULTS_DIR = Path(__file__).parent / "results"

def _load_dataset() -> list[dict]:
    dataset = []
    with open(_DATASET, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and JavaScript-style comments
            if not line or line.startswith("//"):
                continue
            dataset.append(json.loads(line))
    return dataset


def _flatten_context(worker_results) -> str:
    # Safely extract and join all text chunk content across workers
    if not worker_results:
        return ""
        
    contexts = []
    for res in worker_results:
        # Duck-typing to handle both dicts and objects
        chunks = res.get("chunks", []) if isinstance(res, dict) else getattr(res, "chunks", [])
        
        for chunk in chunks:
            content = chunk.get("content", "") if isinstance(chunk, dict) else getattr(chunk, "content", str(chunk))
            contexts.append(content)
            
    return "\n\n".join(contexts)


async def run_eval(strategy: str, provider: str, out_name: str):
    # Ensure results directory exists
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    #  Load dataset & setup orchestrator
    dataset = _load_dataset()
    print(f"Loaded {len(dataset)} questions from dataset.")
    
    orch = Orchestrator(provider=provider, retrieval_strategy=strategy)
    if hasattr(orch, "setup"):
        await orch.setup()
        
    # Setup the fixed yardstick Judge LLM (Always use Gemini for grading to keep scores comparable)
    judge = LLMClient(provider="gemini")
    
    results = []
    
    #Loop through dataset
    for idx, row in enumerate(dataset):
        print(f"[{idx+1}/{len(dataset)}] Evaluating: {row['question']}")
        
        start_time = time.time()
        
        try:
            # Run your agent!
            res = await orch.run(row["question"])
        
            # Adjust these extractions based on exactly what your current agent returns
            answer = getattr(res.report, "text", res.get("answer", "")) if not isinstance(res, str) else res
            sources = getattr(res.report, "sources", res.get("sources", [])) if not isinstance(res, str) else []
            context = _flatten_context(getattr(res, "worker_results", res.get("worker_results", [])))
            cost_usd = getattr(res, "cost_usd", res.get("cost_usd", 0.0)) if not isinstance(res, str) else 0.0
            total_tokens = getattr(res, "total_tokens", res.get("total_tokens", 0)) if not isinstance(res, str) else 0
          
            
        except Exception as e:
            print(f"  Agent crashed on question: {e}")
            answer, sources, context, cost_usd, total_tokens = "", [], "", 0.0, 0
            
        latency = time.time() - start_time
        
        # Calculate metrics
        cit_acc = citation_accuracy(sources, row.get("must_cite_sources", []))
        faith_score, unsupported = await faithfulness(answer, context, judge)
        rel_score = await answer_relevance(row["question"], answer, judge)
        hallucination_rate = 1.0 - faith_score if answer else 1.0
        
        # Save individual result
        results.append({
            "question": row["question"],
            "metrics": {
                "citation_accuracy": cit_acc,
                "faithfulness": faith_score,
                "answer_relevance": rel_score,
                "hallucination_rate": hallucination_rate
            },
            "performance": {
                "latency_sec": round(latency, 2),
                "cost_usd": cost_usd,
                "total_tokens": total_tokens
            },
            "outputs": {
                "answer": answer,
                "unsupported_claims": unsupported,
                "sources_cited": sources
            }
        })
        print(f"  ↳ Scores - Rel: {rel_score:.2f} | Faith: {faith_score:.2f} | Citations: {cit_acc:.2f}")

    # Aggregate Data
    if results:
        agg = {
            "mean_citation_accuracy": round(mean(r["metrics"]["citation_accuracy"] for r in results), 3),
            "mean_faithfulness": round(mean(r["metrics"]["faithfulness"] for r in results), 3),
            "mean_answer_relevance": round(mean(r["metrics"]["answer_relevance"] for r in results), 3),
            "mean_hallucination_rate": round(mean(r["metrics"]["hallucination_rate"] for r in results), 3),
            "mean_latency_sec": round(mean(r["performance"]["latency_sec"] for r in results), 2),
            "total_cost_usd": round(sum(r["performance"]["cost_usd"] for r in results), 4)
        }
    else:
        agg = {}

    #Write to file
    final_output = {
        "config": {
            "strategy": strategy,
            "provider": provider
        },
        "aggregate": agg,
        "results": results
    }
    
    out_path = _RESULTS_DIR / out_name
    with open(out_path, "w") as f:
        json.dump(final_output, f, indent=2)
        
    print("\n" + "="*40)
    print(f"Evaluation Complete! Saved to {out_path.name}")
    print(json.dumps(agg, indent=2))
    print("="*40 + "\n")
    
    # Teardown
    if hasattr(orch, "close"):
        await orch.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="baseline", choices=["baseline", "dense_only", "hybrid", "rerank"])
    ap.add_argument("--provider", default="gemini", choices=["gemini", "ollama"])
    ap.add_argument("--out", default="v1_baseline.json")
    args = ap.parse_args()
    
    asyncio.run(run_eval(args.strategy, args.provider, args.out))


if __name__ == "__main__":
    main()