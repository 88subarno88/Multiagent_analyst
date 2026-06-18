# Deep Research Agent with Memory & Evals

A multi-agent research engine. You give it a question; it breaks the question
into smaller sub-questions, decides what it already knows versus what it needs to
look up, searches the web in parallel, remembers what it learns, writes a single
answer with citations, and then **scores its own answers** against a labelled
test set so the quality is a real measured number, not a guess.

The point of this project is not just "it works." The point is that it is run as
an experiment: several retrieval strategies are each scored on the same 50
questions, and the results are compared in a table so you can see what each
upgrade actually bought.

---

## Headline result

Adding **hybrid retrieval** (vector search + keyword search, fused together)
raised faithfulness from **0.30 to 0.47** and cut the hallucination rate from
**0.70 to 0.53** compared to a plain vector-search baseline.

Adding a **cross-encoder reranker** on top did **not** improve quality on this
set of conceptual questions, and it made each query about **2.4× slower**. That
is a real, measured finding (explained in the report section below), not a bug.

The whole pipeline runs **fully locally** on a single laptop GPU with a local
model (qwen2.5:7b via Ollama), so the **API cost is effectively zero**.

---

## Benchmark

All three configurations were scored on the same 50-question labelled set, using
the same local model (`qwen2.5:7b`) for both the agent and the judge.

| Strategy            | Faithfulness | Answer Relevance | Citation Accuracy | Hallucination Rate | Avg Latency (s) |
|---------------------|:------------:|:----------------:|:-----------------:|:------------------:|:---------------:|
| `dense_only` (v2)   | 0.301        | 0.192            | 0.66              | 0.699              | 113             |
| `hybrid` (v3)       | **0.467**    | **0.360**        | 0.64              | **0.533**          | **97**          |
| `rerank` (v4)       | 0.458        | 0.348            | 0.66              | 0.542              | 230             |

Higher is better for everything **except** Hallucination Rate and Latency, where
lower is better. The raw per-question results are stored in `evals/results/`.

**Best configuration: hybrid retrieval.** It had the highest faithfulness and
relevance, the lowest hallucination, and was actually the fastest of the three.

---

## What each metric means (in plain words)

- **Faithfulness** — Of the facts in the answer, how many are actually backed by
  the sources the system retrieved? High = the model is not making things up.
- **Answer Relevance** — Does the answer actually respond to the question that
  was asked? High = on-topic and complete.
- **Citation Accuracy** — Do the sources the answer cites match the sources we
  hand-labelled as the correct ones for that question?
- **Hallucination Rate** — The opposite of faithfulness (1 − faithfulness). The
  share of claims with no support.
- **Latency** — How long one question takes end to end.

---

## How the system works (the four "planes")

The system is built in four parts. Think of them as four jobs.

```
                 ┌──────────────────────────────────────────────┐
 your question → │              ORCHESTRATION PLANE              │
                 │                                                │
                 │   PLANNER ── breaks the question into ──┐      │
                 │                sub-questions            ▼      │
                 │                              ┌────────────────┐ │
                 │   SYNTHESIZER ◄───────────── │ WORKER agents  │ │
                 │   (writes the cited report)  │ run in parallel│ │
                 │                              └───┬────────┬───┘ │
                 └──────────────────────────────────┼────────┼─────┘
                                                     ▼        ▼
                  ┌───────────────────┐   ┌────────────────────────┐
                  │   MEMORY PLANE    │   │     TOOLING PLANE       │
                  │  chunk → embed →  │◄──│  Tavily / Wikipedia /   │
                  │  pgvector store → │   │  arXiv / scrape + Jina  │
                  │  semantic search  │   └────────────────────────┘
                  └───────────────────┘
                            ▲
                            │  everything above runs INSIDE the harness below
        ┌───────────────────┴───────────────────────────────────────┐
        │  EVALUATION PLANE: test set → run pipeline → score          │
        │  (faithfulness, relevance, citation accuracy, hallucination,│
        │   latency) → results/*.json                                 │
        └─────────────────────────────────────────────────────────────┘
```

**1. Orchestration plane — who does what.**
- The **planner** turns your question into 3–6 smaller sub-questions, and notes
  which sub-questions depend on others.
- **Worker** agents each take one sub-question and run in parallel. Independent
  sub-questions run at the same time; ones that depend on an earlier answer wait
  their turn.
- The **synthesizer** takes everything the workers found and writes one final
  report, with inline citations like `[1]`, `[2]`, and is told to only state
  facts that appear in the retrieved text.

**2. Memory plane — what the system remembers (this is the RAG part).**
- Scraped pages are split into ~500-word **chunks** (with a small overlap so a
  fact split across a boundary is not lost).
- Each chunk is turned into a **vector** (a list of numbers representing its
  meaning) using a local embedding model, and stored in **pgvector** (a vector
  database that lives inside Postgres).
- Before a worker hits the web, it first **checks memory**. If it already has
  good-enough chunks, it skips searching entirely. This is the key efficiency
  win over a naive "always re-search" agent.

**3. Tooling plane — how it reaches the outside world.**
- **Tavily** for web search (and it can return cleaned page content directly).
- **Wikipedia** and **arXiv** as free, reliable sources that never block us —
  great for the technical/conceptual questions in this project.
- A **scraper** (httpx + BeautifulSoup) for normal pages, with a **Jina Reader**
  fallback that rescues pages which block direct scraping.

**4. Evaluation plane — proving it works.**
- Wraps the whole pipeline. For each question in the test set it runs the agent,
  then a "judge" model scores the answer on faithfulness and relevance, and a
  deterministic check scores citation accuracy.
- Writes everything to a versioned JSON file in `evals/results/`.

---

## How one question flows through the system

1. **Plan.** The planner splits the question into sub-questions.
2. **Check memory first.** Each sub-question is embedded and searched against the
   vector store. If strong matches already exist, reuse them — no web call.
3. **Fetch what's missing.** Otherwise: search (Tavily) → gather content from
   Tavily/Wikipedia/arXiv (scrape only when needed) → chunk → embed → write back
   into memory so next time is cheaper.
4. **Grade and correct.** A lightweight LLM call grades whether the retrieved
   context actually answers the sub-question. If it's weak, the worker rewrites
   the query and retries (a bounded number of times so it can't loop forever).
   This is the "corrective RAG" idea.
5. **Synthesize.** The synthesizer writes the final cited report using only the
   retrieved context.
6. **Evaluate.** When run through the harness, the answer is scored and saved.

---

## File structure

```
deep-research-agent/
├── README.md                  # this file
├── requirements.txt           # Python dependencies
├── .env.example               # which keys to set (copy to .env)
├── docker-compose.yml         # one command to start Postgres + pgvector
│
├── src/
│   ├── config.py              # all settings in one place (models, keys, knobs)
│   ├── orchestrator.py        # the "brain": plan → run workers → synthesize
│   ├── observability.py       # cost tracking, timing, per-query budget
│   │
│   ├── models/
│   │   ├── llm.py             # talks to Gemini or Ollama (one interface)
│   │   └── embeddings.py      # local sentence-transformers embedder
│   │
│   ├── agents/
│   │   ├── planner.py         # question → sub-questions + dependency graph
│   │   ├── worker.py          # one sub-question → researched context
│   │   ├── synthesizer.py     # all context → one cited report
│   │   └── prompts/           # the system prompts, as plain .txt files
│   │       ├── planner.txt
│   │       ├── worker_grader.txt
│   │       ├── synthesizer.txt
│   │       └── query_rewrite.txt
│   │
│   ├── memory/
│   │   ├── chunker.py         # splits text into overlapping chunks
│   │   ├── store.py           # pgvector: save chunks, vector + keyword search
│   │   ├── retrieval.py       # the 3 strategies: dense / hybrid / rerank
│   │   └── schema.sql         # database tables (sources, documents, chunks)
│   │
│   ├── tools/
│   │   ├── search.py          # Tavily + Wikipedia + arXiv
│   │   └── scrape.py          # web scraper with Jina Reader fallback
│
├── evals/
│   ├── dataset.jsonl          # the 50 labelled questions
│   ├── metrics.py             # faithfulness, relevance, citation accuracy
│   ├── run_eval.py            # runs the pipeline over the dataset, scores it
│   └── results/               # versioned scores: v2_rag.json, v3_hybrid.json...
│
├── app/
│   └── streamlit_app.py       # simple UI to watch the agent + see eval results
│
└── tests/
    └── test_chunker.py        # unit tests for the deterministic chunker
```

### What the most important files do

- **`src/config.py`** — every setting lives here so experiments are repeatable:
  which model to use, chunk size, how many chunks to retrieve, the per-query
  budget, etc.
- **`src/orchestrator.py`** — ties everything together. Runs the planner, then
  the workers (in dependency order), then the synthesizer.
- **`src/memory/retrieval.py`** — the heart of the benchmark. It implements the
  three strategies you can switch between:
  - `dense_only`: vector search only.
  - `hybrid`: vector search + keyword search, fused with Reciprocal Rank Fusion.
  - `rerank`: hybrid, then a cross-encoder re-scores the shortlist.
- **`src/models/llm.py`** — one wrapper that can talk to either Gemini (cloud) or
  Ollama (local), so the rest of the code doesn't care which model is used.
- **`evals/run_eval.py`** — the measuring stick. Run it to score the system.

---

## Tech stack

| Layer            | Choice                                              |
|------------------|-----------------------------------------------------|
| Language         | Python 3.11+, `asyncio`                             |
| LLM (agent+judge)| `qwen2.5:7b` via **Ollama** (local, free)           |
| LLM (alternative)| Gemini 2.5 Flash (cloud) — supported via `--provider gemini` |
| Embeddings       | `all-MiniLM-L6-v2` (sentence-transformers, local, 384-dim) |
| Reranker         | `bge-reranker-base` cross-encoder (local)           |
| Vector store     | **pgvector** on Postgres                            |
| Web tools        | Tavily, Wikipedia API, arXiv API, httpx + BeautifulSoup, Jina Reader |
| Evaluation       | custom LLM-as-judge + deterministic citation check  |
| UI               | Streamlit                                           |

---

## How to run it

**1. Install dependencies** (a Python virtual environment is recommended):
```bash
pip install -r requirements.txt
```

**2. Set up your keys.** Copy the example file and fill it in:
```bash
cp .env.example .env
```
`.env` needs:
```
TAVILY_API_KEY=your_tavily_key
DATABASE_URL=postgresql://research:research@localhost:5432/research
# Optional, only if you use --provider gemini:
GEMINI_API_KEY=your_gemini_key
```

**3. Start the database** (Postgres + pgvector, one command):
```bash
docker compose up -d
```

**4. Install a local model** (only needed for the local/free path):
```bash
ollama pull qwen2.5:7b
```

**5. Try a single question:**
```bash
python -m src.orchestrator "What is Reciprocal Rank Fusion?"
```

**6. Run the full benchmark** (one command per strategy):
```bash
python -m evals.run_eval --strategy dense_only --provider ollama --out v2_rag.json
python -m evals.run_eval --strategy hybrid     --provider ollama --out v3_hybrid.json
python -m evals.run_eval --strategy rerank     --provider ollama --out v4_rerank.json
```

**7. Optional — open the UI:**
```bash
streamlit run app/streamlit_app.py
```

---

# Technical Report

*What I tried, what moved the number, and what surprisingly didn't.*

## The setup

I evaluated the system on a hand-labelled set of **50 research questions** about
RAG, retrieval, and agent topics. Each question has a short ground-truth answer
and a list of sources that *should* be cited. Four metrics were computed per
question: faithfulness and answer relevance (scored by an LLM judge),
citation accuracy (a deterministic domain-match check), and hallucination rate
(1 − faithfulness).

Everything ran **locally** on a laptop with an RTX 4050 (6 GB VRAM): the agent
and the judge both used `qwen2.5:7b` through Ollama, embeddings used a local
sentence-transformers model, and the vector store was pgvector in Postgres.
**No paid API calls were used in the final benchmark.**

## What I compared

Three retrieval strategies, each scored on the same 50 questions:

1. **`dense_only`** — plain vector (semantic) search. This is the baseline that
   most simple RAG demos stop at.
2. **`hybrid`** — vector search **plus** keyword search, with the two ranked
   lists fused using Reciprocal Rank Fusion (RRF). The idea: vector search
   catches *meaning/paraphrase*, keyword search catches *exact terms and names*
   that vector search can miss.
3. **`rerank`** — start from hybrid, over-fetch 20 candidates, then use a
   cross-encoder reranker to re-score and keep the best 5. A cross-encoder reads
   the question and a chunk *together*, so it judges relevance more accurately
   than the first-pass search — but it's slower.

## What moved the number

**Hybrid retrieval was the big win.** Going from `dense_only` to `hybrid`:

- Faithfulness rose **0.30 → 0.47** (a 55% relative jump).
- Answer relevance rose **0.19 → 0.36** (nearly doubled).
- Hallucination dropped **0.70 → 0.53**.
- And it was actually slightly *faster* (113s → 97s).

The reason this makes sense: many of the questions contain exact technical terms
(BM25, HNSW, RRF, pgvector). Pure vector search sometimes retrieves chunks that
are *about the general topic* but miss the chunk that names the exact term. The
keyword arm catches those exact-term matches, and RRF lets the two methods vote
on the final ranking. Better chunks in → a more grounded answer out.

## What surprisingly didn't

**The cross-encoder reranker did not help here — and it cost a lot of time.**
Going from `hybrid` to `rerank`:

- Faithfulness slightly *dropped* (0.467 → 0.458).
- Relevance slightly *dropped* (0.360 → 0.348).
- Latency went **up 2.4×** (97s → 230s).

This was the most interesting result, because reranking is usually described as
"the single highest-ROI retrieval upgrade." On this workload it wasn't, and I
think the reasons are:

1. **The questions are mostly conceptual/definitional.** Hybrid already surfaces
   the right chunk; there isn't a long, noisy candidate list for the reranker to
   rescue.
2. **The corpus is small and fairly clean.** Reranking shines when you over-fetch
   from a large, messy pool. With a small memory store, the top hybrid results
   are already good, so re-ordering them barely changes the final 5.
3. **The reranker model is small** (`bge-reranker-base`), so its precision gain
   was not enough to beat the ordering RRF already produced.

The honest takeaway: **reranking is not free, and it is not always worth it.** On
a large, noisy corpus it would likely pay off; on this small conceptual set it
added latency for no quality gain.

## Cost and quality tradeoff

The whole system was designed to be swappable between a cloud model (Gemini) and
a local model (Ollama). During development the cloud free tiers ran out (rate
limits), so the final benchmark used a fully local model. The result is a
pipeline with **zero API cost**, at the price of **higher latency** (around 1–2
minutes per question on a 6 GB laptop GPU) and a **weaker judge and writer** than
a frontier cloud model would provide.

## Limitations (being honest)

- The **judge is a local 7B model**, which is noisier and more lenient than a
  frontier model. The absolute scores should be read as rough; the *relative*
  comparison across strategies is the trustworthy part, since every strategy was
  scored by the same judge.
- The 7B model is also a **weaker writer**, which is part of why relevance scores
  are modest.
- **Citation accuracy is roughly flat (~0.65)** across strategies, because the
  retrieval *method* doesn't change much about *which sources* end up cited.
- The **memory store was shared across the three runs**, so each strategy
  retrieved from the same accumulated corpus (a fair comparison of retrieval
  method, but not a from-scratch corpus per run).
- Some web sources block scraping; the system degrades gracefully by falling back
  to Wikipedia, arXiv, and a Jina Reader proxy, but a few pages are still missed.

## Engineering notes

A few things that were real work beyond the core RAG logic:

- **Provider-agnostic LLM client** with key rotation and exponential backoff on
  rate-limit (429) errors, so a long run survives a flaky free tier.
- **Graceful degradation** everywhere: a blocked scrape, an exhausted search
  quota, or a failed source never crashes a whole question — the system falls
  back to other sources and keeps going.
- **Local-inference tuning**: fitting a 7B model on 6 GB of VRAM required tuning
  the context window and using flash attention; too large a context pushed the
  model onto the CPU and made it ~10× slower.
- **Cost/latency tracking** on every run, with a per-query token budget that
  stops a runaway query cleanly instead of crashing.

## Conclusion

The measured conclusion is simple and defensible: **hybrid retrieval was a clear
win over plain vector search (faithfulness 0.30 → 0.47), while a cross-encoder
reranker was not worth its cost on this conceptual-question set.** Being able to
show both a positive result and a well-explained negative one — with the numbers
to back them — is the real output of this project.
