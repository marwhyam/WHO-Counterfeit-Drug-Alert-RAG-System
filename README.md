# WHO Counterfeit Drug Alert RAG System

A multilingual retrieval-augmented generation system over WHO Medical Product Alerts. Lets analysts query 80 real WHO alerts in any language and receive cross-verified, cited answers about falsified and substandard medicines globally.

---

## What it does

WHO, INTERPOL, and FDA publish counterfeit drug alerts in fragmented formats across different languages and regions. This system ingests all public WHO Medical Product Alerts (2013-2026) and provides a unified intelligence interface where an analyst can ask:

- Which falsified cancer medicines were found in Africa?
- What contaminated paediatric medicines caused child deaths?
- Which drugs were falsified in Pakistan between 2019 and 2024?
- (in Urdu) جعلی دوائیں پاکستان میں کون سی ہیں؟

And receive grounded, cited answers backed by real WHO data.

---

## Architecture

```
User Query (any language)
        |
   DELTA Layer
   Multilingual query fusion
   Detects language, fuses native + English embeddings
   arxiv 2601.02956
        |
   Cross-lingual Vector Retrieval
   Qdrant + paraphrase-multilingual-mpnet-base-v2
   arxiv 2505.10089 (XRAG benchmark)
        |
   T-GRAG Temporal Knowledge Graph
   Time-stamped nodes and edges
   Tracks how drug incidents evolve over time
   UPDATED_BY edges link same drug across alerts
   arxiv 2508.01680
        |
   Hyper-RAG Verification
   Hyperedges connect alerts sharing the same entity
   Facts confirmed by 2+ sources get HIGH confidence
   Facts from 1 source are flagged as hallucination risk
   arxiv 2504.08758 (Nature Communications 2026)
        |
   Grounded Cited Answer
   TinyLlama via Ollama (fully local, no API key)
   React dashboard via FastAPI
```

---

## Research Papers

| Paper | ID | Venue |
|---|---|---|
| XRAG: Cross-lingual Retrieval-Augmented Generation | arxiv 2505.10089 | EMNLP 2025 |
| T-GRAG: Temporal GraphRAG Framework | arxiv 2508.01680 | August 2025 |
| Hyper-RAG: Hypergraph-Driven RAG | arxiv 2504.08758 | Nature Communications 2026 |
| DELTA: Debiased Multilingual RAG | arxiv 2601.02956 | January 2026 |

---

## Data

All data is sourced from publicly available WHO Medical Product Alerts.

Source: https://www.who.int/teams/regulation-prequalification/incidents-and-SF/full-list-of-who-medical-product-alerts

80 alerts spanning 2013 to 2026 covering falsified and substandard medicines across all WHO regions. The downloader script regenerates the dataset automatically.

---

## Setup

**Requirements:**
- Python 3.10+
- Ollama installed from https://ollama.com

**Install dependencies:**
```bash
pip install qdrant-client sentence-transformers requests beautifulsoup4 fastapi uvicorn networkx langdetect
```

**Pull the local LLM:**
```bash
ollama pull tinyllama
```

**Step 1 - Download WHO alerts:**
```bash
python who_downloader_v2.py
```

**Step 2 - Enrich and fetch full content:**
```bash
python who_downloader_v2.py
```

**Step 3 - Build vector index:**
```bash
python ingest_pipeline_v2.py
```

**Step 4 - Run the full pipeline (optional CLI):**
```bash
python rag_query_pipeline_tiny.py
python temporal_graph.py
python hyper_rag.py
python delta_layer.py
```

**Step 5 - Launch dashboard:**
```bash
python dashboard_backend.py
```

Open browser at http://localhost:8000

---

## Files

| File | Description |
|---|---|
| who_alert_downloader.py | Scrapes WHO alert list and saves metadata |
| who_downloader_v2.py | Fetches full alert content and extracts entities |
| ingest_pipeline_v2.py | Chunks, embeds, and indexes alerts into Qdrant |
| rag_query_pipeline_tiny.py | Baseline RAG with TinyLlama |
| temporal_graph.py | T-GRAG temporal knowledge graph layer |
| hyper_rag.py | Hyper-RAG hyperedge verification layer |
| delta_layer.py | DELTA multilingual debiasing layer |
| dashboard_backend.py | FastAPI backend + full React dashboard |

---

## Dashboard Views

**Search Intelligence** - Query in any language, see retrieved alerts with relevance scores, Hyper-RAG verification panel showing cross-verified facts, and a grounded LLM answer.

**Drug Timeline** - Enter any drug name to see its full alert history plotted chronologically across countries. Powered by T-GRAG temporal graph.

**Alert Database** - Full table of all 80 WHO alerts with type, date, and country metadata.

**Statistics** - Alert type breakdown, most affected countries bar chart, and system metrics.

---

## Languages Supported

English, Urdu, Arabic, French, Spanish, and 45+ others via the multilingual embedding model. The DELTA layer handles query fusion across languages to prevent English bias in retrieval.

---

## Stack

- Vector store: Qdrant
- Embeddings: sentence-transformers paraphrase-multilingual-mpnet-base-v2
- Knowledge graph: NetworkX
- LLM: TinyLlama via Ollama (local, no API key required)
- Backend: FastAPI
- Dashboard: Vanilla JS + HTML served via FastAPI
