import asyncio
import json
import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                HRFlowable, ListFlowable, ListItem, Table,
                                TableStyle)

from src.orchestrator import Orchestrator

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "evals" / "results"

ACCENT = colors.HexColor("#2563eb")
DARK = colors.HexColor("#1e293b")
GREY = colors.HexColor("#475569")

st.set_page_config(page_title="Deep Research Agent", layout="wide")
st.title("Deep Research Agent — Memory & Evals")

tab_run, tab_evals = st.tabs(["Ask a question", "Benchmark results"])


# PDF helpers

def _pdf_styles():
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], textColor=DARK,
                                fontSize=22, spaceAfter=4, leading=26),
        "qmeta": ParagraphStyle("q", parent=ss["Normal"], textColor=GREY,
                                fontSize=11, spaceAfter=14, leading=15),
        "h1": ParagraphStyle("h1", parent=ss["Heading1"], textColor=ACCENT,
                            fontSize=16, spaceBefore=14, spaceAfter=6, leading=20),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], textColor=DARK,
                            fontSize=13, spaceBefore=10, spaceAfter=5, leading=17),
        "h3": ParagraphStyle("h3", parent=ss["Heading3"], textColor=DARK,
                            fontSize=11.5, spaceBefore=8, spaceAfter=4, leading=15),
        "body": ParagraphStyle("b", parent=ss["Normal"], textColor=colors.HexColor("#0f172a"),
                            fontSize=10.5, spaceAfter=7, leading=15),
        "bullet": ParagraphStyle("bul", parent=ss["Normal"], textColor=colors.HexColor("#0f172a"),
                            fontSize=10.5, leading=15),
        "src": ParagraphStyle("s", parent=ss["Normal"], textColor=GREY,
                            fontSize=9, leading=13, spaceAfter=3),
    }


def _clean_inline(text):
    text = re.sub(r"\\\[.*?\\\]", " (formula omitted) ", text, flags=re.DOTALL)
    text = re.sub(r"\\\(.*?\\\)", " (formula) ", text, flags=re.DOTALL)
    text = re.sub(r"\$\$.*?\$\$", " (formula omitted) ", text, flags=re.DOTALL)
    text = re.sub(r"\$[^$]+\$", " (formula) ", text)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', text)
    return text.strip()


def markdown_to_pdf(question, md_text, sources):
    S = _pdf_styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="Research Report",
                            leftMargin=22*mm, rightMargin=22*mm,
                            topMargin=20*mm, bottomMargin=18*mm)
    story = [Paragraph("Research Report", S["title"]),
             HRFlowable(width="100%", thickness=2, color=ACCENT, spaceAfter=10),
             Paragraph(f"Question: {question}", S["qmeta"])]

    bullets = []

    def flush():
        nonlocal bullets
        if bullets:
            items = [ListItem(Paragraph(b, S["bullet"]), leftIndent=6) for b in bullets]
            story.append(ListFlowable(items, bulletType="bullet",
                                      bulletColor=ACCENT, leftIndent=12))
            story.append(Spacer(1, 4))
            bullets = []

    for raw in md_text.split("\n"):
        line = raw.rstrip()
        if not line.strip():
            flush(); continue
        if line.lstrip().startswith("---"):
            flush()
            story.append(HRFlowable(width="100%", thickness=0.6,
                                    color=colors.HexColor("#cbd5e1"),
                                    spaceBefore=6, spaceAfter=6))
            continue
        h = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if h:
            flush()
            level = len(h.group(1))
            style = S["h1"] if level <= 1 else (S["h2"] if level == 2 else S["h3"])
            story.append(Paragraph(_clean_inline(h.group(2)), style))
            continue
        b = re.match(r"^\s*[-*+]\s+(.*)$", line)
        if b:
            bullets.append(_clean_inline(b.group(1))); continue
        nb = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if nb:
            bullets.append(_clean_inline(nb.group(1))); continue
        flush()
        story.append(Paragraph(_clean_inline(line), S["body"]))

    flush()

    if sources:
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=1, color=ACCENT, spaceAfter=6))
        story.append(Paragraph("Sources", S["h2"]))
        for i, url in enumerate(sources, 1):
            safe = url.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(f'<b>[{i}]</b> {safe}', S["src"]))

    doc.build(story)
    buf.seek(0)
    return buf.read()


def benchmark_pdf(rows):
    S = _pdf_styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="Benchmark Report",
                            leftMargin=22*mm, rightMargin=22*mm,
                            topMargin=20*mm, bottomMargin=18*mm)
    story = [Paragraph("Deep Research Agent — Benchmark", S["title"]),
             HRFlowable(width="100%", thickness=2, color=ACCENT, spaceAfter=10),
             Paragraph("Three retrieval strategies scored on the same 50-question "
                       "labelled set, using a local qwen2.5:7b model for both the "
                       "agent and the judge.", S["body"]), Spacer(1, 10)]

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
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2ff")]),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 18))
    story.append(Paragraph("Findings", S["h2"]))
    for para in [
        "Hybrid retrieval (dense + BM25 fused with Reciprocal Rank Fusion) was the "
        "clear win over the dense-only baseline, raising faithfulness from 0.30 to "
        "0.47 and cutting the hallucination rate from 0.70 to 0.53, while also "
        "running slightly faster.",
        "Adding a cross-encoder reranker did not improve quality on this conceptual "
        "question set and roughly 2.4x'd the latency — a measured negative result.",
        "The pipeline runs fully locally at zero API cost; the trade-off is higher "
        "latency and a noisier 7B judge, so absolute scores are rough but the "
        "relative comparison across strategies is reliable.",
    ]:
        story.append(Paragraph(para, S["body"]))

    doc.build(story)
    buf.seek(0)
    return buf.read()


def answer_to_md(question, answer, sources):
    out = [f"# Research Report\n", f"**Question:** {question}\n", answer, ""]
    if sources:
        out.append("\n## Sources\n")
        for i, url in enumerate(sources, 1):
            out.append(f"[{i}] {url}")
    return "\n".join(out)


# data

async def _run_agent(question, provider, strategy):
    orch = Orchestrator(provider=provider, retrieval_strategy=strategy)
    await orch.setup()
    try:
        res = await orch.run(question)
        cost = res.cost.get("cost_usd", 0.0) if isinstance(res.cost, dict) else 0.0
        tokens = res.cost.get("total_tokens", 0) if isinstance(res.cost, dict) else 0
        return res.report.text, res.report.sources, cost, tokens, len(res.worker_results)
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
                "file": f.name, "strategy": cfg.get("strategy", "?"),
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


# UI: Ask a question

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
                    _run_agent(question, provider, strategy))
                st.session_state["last"] = {
                    "question": question, "answer": answer, "sources": sources,
                    "cost": cost, "tokens": tokens, "n_workers": n_workers}
            except Exception as e:
                st.session_state.pop("last", None)
                st.error(f"Run failed: {e}")
                st.caption("Check Postgres is up (docker compose up -d) and the model is reachable.")

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
        d1, d2, _ = st.columns([2, 2, 5])
        with d1:
            pdf = markdown_to_pdf(last["question"], last["answer"], last["sources"])
            st.download_button("Download PDF", data=pdf,
                               file_name="research_report.pdf",
                               mime="application/pdf", type="primary",
                               use_container_width=True)
        with d2:
            md = answer_to_md(last["question"], last["answer"], last["sources"])
            st.download_button("Download Markdown", data=md,
                               file_name="research_report.md",
                               mime="text/markdown", type="primary",
                               use_container_width=True)


# UI: Benchmark

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
        b1, b2, _ = st.columns([2, 2, 5])
        with b1:
            st.download_button("Download PDF", data=benchmark_pdf(rows),
                               file_name="benchmark_report.pdf",
                               mime="application/pdf", type="primary",
                               use_container_width=True)
        with b2:
            readme = ROOT / "README.md"
            if readme.exists():
                st.download_button("Download README", data=readme.read_text(),
                                   file_name="README.md", mime="text/markdown",
                                   type="primary", use_container_width=True)
            else:
                st.caption("README.md not found in project root.")