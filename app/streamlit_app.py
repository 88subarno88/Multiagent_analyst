import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from src.orchestrator import Orchestrator

RESULTS_DIR = Path(__file__).parent.parent / "evals" / "results"

st.set_page_config(page_title="Deep Research Agent", layout="wide")
st.title("Deep Research Agent — Memory & Evals")

tab_run, tab_evals = st.tabs(["Ask a question", "Benchmark results"])


async def _run_agent(question: str, provider: str, strategy: str):
    orch = Orchestrator(provider=provider, retrieval_strategy=strategy)
    await orch.setup()
    try:
        res = await orch.run(question)
        answer = res.report.text
        sources = res.report.sources
        cost = res.cost.get("cost_usd", 0.0) if isinstance(res.cost, dict) else 0.0
        tokens = res.cost.get("total_tokens", 0) if isinstance(res.cost, dict) else 0
        n_workers = len(res.worker_results)
        return answer, sources, cost, tokens, n_workers
    finally:
        if hasattr(orch, "store") and hasattr(orch.store, "close"):
            await orch.store.close()


with tab_run:
    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input("Your research question",
                                 "What is Reciprocal Rank Fusion?")
    with col2:
        provider = st.selectbox("Model provider", ["ollama", "gemini"])
        strategy = st.selectbox("Retrieval strategy",
                                ["hybrid", "dense_only", "rerank"])

    if st.button("Research", type="primary"):
        with st.spinner("Planning, retrieving, and synthesizing... this can take a minute or two on a local model."):
            try:
                answer, sources, cost, tokens, n_workers = asyncio.run(
                    _run_agent(question, provider, strategy)
                )
                st.subheader("Report")
                st.markdown(answer)

                if sources:
                    st.subheader("Sources")
                    for i, url in enumerate(sources, 1):
                        st.markdown(f"[{i}] {url}")

                c1, c2, c3 = st.columns(3)
                c1.metric("Workers run", n_workers)
                c2.metric("Tokens", tokens)
                c3.metric("Cost (USD)", f"${cost:.4f}")
            except Exception as e:
                st.error(f"Run failed: {e}")
                st.caption("Check that Postgres is up (docker compose up -d) and "
                           "the model provider is reachable.")


with tab_evals:
    st.subheader("Benchmark — strategies scored on the labelled set")

    files = sorted(RESULTS_DIR.glob("*.json")) if RESULTS_DIR.exists() else []
    if not files:
        st.info("No results yet. Run: python -m evals.run_eval --strategy hybrid "
                "--provider ollama --out v3_hybrid.json")
    else:
        rows = []
        for f in files:
            try:
                data = json.loads(f.read_text())
                agg = data.get("aggregate", data)
                cfg = data.get("config", {})
                rows.append({
                    "file": f.name,
                    "strategy": cfg.get("strategy", "?"),
                    "faithfulness": agg.get("mean_faithfulness"),
                    "relevance": agg.get("mean_answer_relevance"),
                    "citation": agg.get("mean_citation_accuracy"),
                    "hallucination": agg.get("mean_hallucination_rate"),
                    "latency_sec": agg.get("mean_latency_sec"),
                    "cost_usd": agg.get("total_cost_usd"),
                })
            except Exception as e:
                st.warning(f"Could not read {f.name}: {e}")

        if rows:
            st.dataframe(rows, use_container_width=True)
            st.bar_chart(
                {r["file"]: r["faithfulness"] for r in rows if r["faithfulness"] is not None},
            )
            st.caption("Higher faithfulness/relevance/citation is better; "
                       "lower hallucination/latency is better.")
