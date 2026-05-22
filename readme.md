# Multi-Agent Research Analyst

Autonomous competitive intelligence briefs — powered by Gemini AI + Tavily Search.
Type a company name, watch 4 agents work in parallel, download a PDF report.

**Live demo → [multiagentanalyst.streamlit.app](https://multiagentanalyst.streamlit.app/)**

---

## How it works

A boss orchestrator agent plans the research, then fires 3 specialist agents **at the same time**:

| Agent | What it does |
|---|---|
|  Web Agent | Market position, products, competitors |
|  News Agent | Last 30 days — funding, launches, leadership changes |
| Financial Agent | Revenue, growth, profitability, key risks |

All findings get passed to a **synthesiser** that writes the final structured brief and exports it as a PDF.

---

## Stack

- **Gemini 2.5 Flash** — orchestration + synthesis
- **Tavily Search API** — real-time web + news search
- **asyncio** — all agents run concurrently
- **BeautifulSoup + httpx** — full article scraping
- **ReportLab** — PDF export
- **Streamlit** — UI

---

## Setup

**1. Clone**
```bash
git clone https://github.com/88subarno88/Multiagent_analyst
cd Multiagent_analyst
```

**2. Virtual environment**
```bash
python3 -m venv my_project_env
source my_project_env/bin/activate      # Windows: my_project_env\Scripts\activate
```

**3. Install**
```bash
pip install -r requirements.txt
```

**4. Add API keys** — create a `.env` file:
```
GEMINI_API_KEY=your_google_key_here
TAVILY_API_KEY=your_tavily_key_here
```

Get your keys:
- Gemini → https://aistudio.google.com/app/apikey (free)
- Tavily → https://tavily.com (1,000 searches/month free)

---

## Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` → enter a company name → click **Generate Brief**.

> Free tier users: expect ~60 seconds. Built-in rate limit handling keeps it stable.

---

## Project structure

```
Multiagent_analyst/
├── app.py
├── .env
├── requirements.txt
├── prompts/
│   ├── orchestrator.txt
│   ├── researcher.txt
│   └── synthesiser.txt
├── agents/
│   ├── orchastor.py
│   ├── searcher.py
│   ├── newsgiver.py
│   ├── Finantial.py
│   └── synthesizer.py
└── tools/
    ├── tavily_client.py
    ├── web_scraper.py
    └── pdf_generator.py
```

---

## Limitations

- **Free tier quota** — Gemini free tier has a daily request limit. If you hit it, wait 24 hours or upgrade to pay-as-you-go on Google AI Studio.
- **Rate limiting** — running the full pipeline makes 4-6 Gemini calls at once. Built-in retries handle this but expect occasional delays.
- **Paywalled sites** — financial pages behind login walls (Bloomberg, FT, WSJ) can't be scraped. The agent falls back to Tavily snippets instead.
- **Private companies** — limited financial data available for startups with no public filings. Revenue figures will often show as `unknown`.
- **PDF accuracy** — the PDF is generated from Markdown. Complex tables or nested lists may not render perfectly.
- **No memory** — each run is fully independent. The system doesn't remember previous searches or build on past reports.
- **English only** — prompts and output are in English. Non-English company pages may be scraped but not summarised accurately.
- **Tavily free tier** — 1,000 searches/month. One full report uses ~15-20 searches, so the free tier supports roughly 50-60 reports per month.