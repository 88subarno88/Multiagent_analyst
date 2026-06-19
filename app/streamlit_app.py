import asyncio
import json
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from src.orchestrator import Orchestrator

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "evals" / "results"

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


def _load_rows():
    rows = []
    files = sorted(RESULTS_DIR.glob("*.json")) if RESULTS_DIR.exists() else []
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
    return rows


def _answer_to_pdf(question: str, answer: str, sources: list) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="Research Report")
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], spaceAfter=8, leading=14)
    story = [Paragraph("Research Report", styles["Title"]), Spacer(1, 10),
             Paragraph(f"<b>Question:</b> {question}", body), Spacer(1, 12)]
    # Render the answer paragraph by paragraph; escape stray markup chars.
    for para in answer.split("\n"):
        p = para.strip()
        if not p:
            continue
        p = p.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(p, body))
    if sources:
        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>Sources</b>", body))
        for i, url in enumerate(sources, 1):
            safe = url.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(f"[{i}] {safe}", body))
    doc.build(story)
    buf.seek(0)
    return buf.read()


def _answer_to_md(question: str, answer: str, sources: list) -> str:
    out = [f"# Research Report\n", f"**Question:** {question}\n", answer, ""]
    if sources:
        out.append("\n## Sources\n")
        for i, url in enumerate(sources, 1):
            out.append(f"[{i}] {url}")
    return "\n".join(out)


def _build_pdf(rows) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="Deep Research Agent — Report")
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Deep Research Agent — Benchmark Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Three retrieval strategies scored on the same 50-question labelled set, "
        "using a local qwen2.5:7b model for both the agent and the judge. Higher "
        "is better for faithfulness, relevance, and citation; lower is better for "
        "hallucination and latency.", styles["Normal"]))
    story.append(Spacer(1, 16))
    header = ["Strategy", "Faith", "Relevance", "Citation", "Halluc.", "Latency(s)"]
    data = [header]
    for r in rows:
        data.append([
            str(r["strategy"]),
            f'{r["faithfulness"]:.3f}' if r["faithfulness"] is not None else "-",
            f'{r["relevance"]:.3f}' if r["relevance"] is not None else "-",
            f'{r["citation"]:.2f}' if r["citation"] is not None else "-",
            f'{r["hallucination"]:.3f}' if r["hallucination"] is not None else "-",
            f'{r["latency_sec"]:.0f}' if r["latency_sec"] is not None else "-",
        ])
    tbl = Table(data, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 20))
    body = ParagraphStyle("body", parent=styles["Normal"], spaceAfter=8, leading=14)
    story.append(Paragraph("Findings", styles["Heading2"]))
    story.append(Paragraph(
        "Hybrid retrieval (dense + BM25 fused with Reciprocal Rank Fusion) was the "
        "clear win over the dense-only baseline, raising faithfulness from 0.30 to "
        "0.47 and cutting the hallucination rate from 0.70 to 0.53, while also "
        "running slightly faster.", body))
    story.append(Paragraph(
        "Adding a cross-encoder reranker did not improve quality on this set of "
        "conceptual questions and roughly 2.4x'd the latency. With a small, clean "
        "corpus and definitional questions, hybrid already surfaces the right "
        "chunks, so reranking has little to fix. This is a measured negative "
        "result, not a failure.", body))
    story.append(Paragraph(
        "The pipeline runs fully locally at zero API cost; the trade-off is higher "
        "latency and a noisier 7B judge, so absolute scores are rough but the "
        "relative comparison across strategies is reliable.", body))
    doc.build(story)
    buf.seek(0)
    return buf.read()


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
                st.session_state["last"] = {
                    "question": question, "answer": answer, "sources": sources,
                    "cost": cost, "tokens": tokens, "n_workers": n_workers,
                }
            except Exception as e:
                st.session_state.pop("last", None)
                st.error(f"Run failed: {e}")
                st.caption("Check that Postgres is up (docker compose up -d) and "
                           "the model provider is reachable.")

    # Show the last result (persists across reruns) + download buttons.
    if "last" in st.session_state:
        last = st.session_state["last"]
        st.subheader("Report")
        st.markdown(last["answer"])
        if last["sources"]:
            st.subheader("Sources")
            for i, url in enumerate(last["sources"], 1):
                st.markdown(f"[{i}] {url}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Workers run", last["n_workers"])
        c2.metric("Tokens", last["tokens"])
        c3.metric("Cost (USD)", f"${last['cost']:.4f}")

        st.divider()
        st.write("**Download this report**")
        d1, d2 = st.columns(2)
        with d1:
            try:
                pdf = _answer_to_pdf(last["question"], last["answer"], last["sources"])
                st.download_button("Download report (PDF)", data=pdf,
                                   file_name="research_report.pdf",
                                   mime="application/pdf", type="primary")
            except Exception as e:
                st.warning(f"PDF needs reportlab (pip install reportlab). {e}")
        with d2:
            md = _answer_to_md(last["question"], last["answer"], last["sources"])
            st.download_button("Download report (Markdown)", data=md,
                               file_name="research_report.md",
                               mime="text/markdown")


with tab_evals:
    st.subheader("Benchmark — strategies scored on the labelled set")
    rows = _load_rows()
    if not rows:
        st.info("No results yet. Run: python -m evals.run_eval --strategy hybrid "
                "--provider ollama --out v3_hybrid.json")
    else:
        st.dataframe(rows, use_container_width=True)
        chart = {r["file"]: r["faithfulness"] for r in rows
                 if r["faithfulness"] is not None}
        if chart:
            st.bar_chart(chart)
        st.caption("Higher faithfulness/relevance/citation is better; "
                   "lower hallucination/latency is better.")

        st.divider()
        st.write("**Download the benchmark report**")
        b1, b2 = st.columns(2)
        with b1:
            try:
                pdf_bytes = _build_pdf(rows)
                st.download_button("Download benchmark (PDF)", data=pdf_bytes,
                                   file_name="benchmark_report.pdf",
                                   mime="application/pdf", type="primary")
            except Exception as e:
                st.warning(f"PDF needs reportlab (pip install reportlab). {e}")
        with b2:
            readme = ROOT / "README.md"
            if readme.exists():
                st.download_button("Download README (Markdown)",
                                   data=readme.read_text(),
                                   file_name="README.md", mime="text/markdown")
            else:
                st.caption("README.md not found in project root.")