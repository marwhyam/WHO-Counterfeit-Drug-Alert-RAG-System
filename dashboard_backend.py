"""
Counterfeit Drug Intelligence Network - Dashboard Backend
==========================================================
FastAPI backend that serves the React dashboard.
Combines all 4 layers: Vector RAG + T-GRAG + Hyper-RAG + DELTA

Install:
    pip install fastapi uvicorn qdrant-client sentence-transformers requests networkx langdetect

Run:
    python dashboard_backend.py
    Then open: http://localhost:8000
"""

import os, re, json, uuid
from pathlib import Path
from collections import defaultdict
from typing import List, Optional

import requests as http_requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# ── Config ────────────────────────────────────────────────────────────────────
ALERTS_DIR      = "who_alerts_v2"
COLLECTION_NAME = "counterfeit_drug_alerts"
EMBED_MODEL     = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_MODEL    = "tinyllama"
TIMEOUT         = 300

app = FastAPI(title="Counterfeit Drug Intelligence Network")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Global state ──────────────────────────────────────────────────────────────
embedder     = None
qdrant       = None
temporal_graph = None
hyperedge_idx  = None
all_alerts     = []

# ── Models ────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    use_temporal: bool = True
    use_hyper: bool = True
    use_delta: bool = True

# ── Parsers ───────────────────────────────────────────────────────────────────
def parse_alert_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    def field(name):
        m = re.search(rf"^{name}:\s*(.+)$", content, re.MULTILINE)
        return m.group(1).strip() if m else ""

    parts = content.split("=" * 60)
    body  = parts[1].strip() if len(parts) > 1 else content

    year = None
    m = re.search(r"(\d{4})", field("DATE"))
    if m:
        year = int(m.group(1))

    month = None
    months = {"january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
              "july":7,"august":8,"september":9,"october":10,"november":11,"december":12}
    for mname, mnum in months.items():
        if mname in field("DATE").lower():
            month = mnum
            break

    return {
        "alert_number": field("ALERT NUMBER"),
        "drug_name":    field("DRUG NAME"),
        "alert_type":   field("ALERT TYPE"),
        "date":         field("DATE"),
        "year":         year,
        "month":        month,
        "countries":    [c.strip() for c in field("COUNTRIES").split(",") if c.strip()],
        "who_regions":  [r.strip() for r in field("WHO REGIONS").split(",") if r.strip()],
        "source_url":   field("SOURCE URL"),
        "body":         body[:1000],
        "filename":     os.path.basename(filepath),
    }

# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global embedder, qdrant, all_alerts, hyperedge_idx, temporal_graph

    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("Connecting to Qdrant...")
    qdrant = QdrantClient(path="qdrant_storage")

    print("Loading alerts...")
    txt_files = sorted(Path(ALERTS_DIR).glob("*.txt"))
    all_alerts = [parse_alert_file(str(fp)) for fp in txt_files]
    print(f"Loaded {len(all_alerts)} alerts")

    print("Building hyperedge index...")
    hyperedge_idx = build_hyperedge_index(all_alerts)

    print("Building temporal graph...")
    temporal_graph = build_temporal_graph_data(all_alerts)

    print("Ready.")

# ── Hyperedge Index ───────────────────────────────────────────────────────────
def build_hyperedge_index(alerts):
    entity_to_alerts = defaultdict(set)
    alert_to_entities = defaultdict(set)

    for alert in alerts:
        num = alert["alert_number"]
        if not num:
            continue
        entities = set()
        drug = alert.get("drug_name","").lower().strip()
        if drug and drug != "unknown":
            core = re.sub(r"\s*\(.*?\)","",drug).strip()
            if len(core) > 2:
                entities.add(f"DRUG:{core}")
        for c in alert.get("countries",[]):
            entities.add(f"COUNTRY:{c.lower()}")
        atype = alert.get("alert_type","")
        if atype:
            entities.add(f"TYPE:{atype}")
        for kw in ["malaria","vaccine","paediatric","children","cancer","contaminated","covid","hepatitis","syrup","fentanyl"]:
            if kw in alert.get("body","").lower():
                entities.add(f"KEYWORD:{kw}")
        for e in entities:
            entity_to_alerts[e].add(num)
            alert_to_entities[num].add(e)

    hyperedges = {}
    he_id = 0
    for entity, alert_set in entity_to_alerts.items():
        if len(alert_set) >= 2:
            hid = f"HE_{he_id}"
            hyperedges[hid] = {"entity": entity, "alerts": list(alert_set), "strength": len(alert_set)}
            he_id += 1

    return {"entity_to_alerts": dict(entity_to_alerts), "hyperedges": hyperedges, "alert_to_entities": dict(alert_to_entities)}

def get_verification(alert_numbers, idx):
    alert_set = set(alert_numbers)
    shared = defaultdict(list)
    for hid, he in idx["hyperedges"].items():
        overlap = alert_set & set(he["alerts"])
        if len(overlap) >= 2:
            for e in [he["entity"]]:
                shared[e].extend(list(overlap))
    verified = []
    for entity, confirming in shared.items():
        count = len(set(confirming))
        verified.append({
            "entity": entity,
            "confirmed_by": list(set(confirming)),
            "confidence": "HIGH" if count >= 2 else "LOW",
            "source_count": count,
        })
    return sorted(verified, key=lambda x: x["source_count"], reverse=True)

# ── Temporal Graph ────────────────────────────────────────────────────────────
def build_temporal_graph_data(alerts):
    drug_timeline = defaultdict(list)
    country_counts = defaultdict(int)
    yearly_counts  = defaultdict(int)

    for alert in alerts:
        drug = alert.get("drug_name","").strip()
        year = alert.get("year")
        if drug and drug != "unknown" and year:
            drug_timeline[drug.lower()].append({
                "alert_number": alert["alert_number"],
                "date": alert["date"],
                "year": year,
                "month": alert.get("month"),
                "alert_type": alert.get("alert_type",""),
                "countries": alert.get("countries",[]),
                "drug_name": drug,
            })
        for country in alert.get("countries",[]):
            country_counts[country] += 1
        if year:
            yearly_counts[year] += 1

    # Sort timelines
    for drug in drug_timeline:
        drug_timeline[drug].sort(key=lambda x: (x["year"], x.get("month") or 0))

    return {
        "drug_timeline": dict(drug_timeline),
        "country_counts": dict(sorted(country_counts.items(), key=lambda x: x[1], reverse=True)),
        "yearly_counts": dict(sorted(yearly_counts.items())),
    }

# ── DELTA query fusion ────────────────────────────────────────────────────────
TRANSLATIONS = {
    "جعلی دوائیں": "falsified medicines", "جعلی ادویات": "counterfeit drugs",
    "پاکستان": "Pakistan", "ویکسین": "vaccine", "بچوں کی دوائیں": "paediatric medicines",
    "médicaments falsifiés": "falsified medicines", "contaminés": "contaminated",
    "vaccins falsifiés": "falsified vaccines", "Afrique": "Africa",
    "أدوية مزيفة": "falsified medicines", "لقاح مزيف": "falsified vaccine",
    "باكستان": "Pakistan", "أفريقيا": "Africa",
}

def detect_lang(text):
    try:
        from langdetect import detect
        return detect(text)
    except:
        if any('\u0600' <= c <= '\u06FF' for c in text):
            return "ar"
        return "en"

def delta_fuse(query, alpha=0.6):
    lang = detect_lang(query)
    translated = query
    for k, v in TRANSLATIONS.items():
        if k in query:
            translated = translated.replace(k, v)
    en_vec  = embedder.encode(translated)
    nat_vec = embedder.encode(query)
    fused   = alpha * en_vec + (1 - alpha) * nat_vec
    norm    = (fused ** 2).sum() ** 0.5
    if norm > 0:
        fused = fused / norm
    return fused.tolist(), lang, translated

# ── Ollama ────────────────────────────────────────────────────────────────────
def ask_ollama(query, context):
    prompt = f"""You are a WHO pharmaceutical intelligence analyst.
Answer using ONLY these sources. Cite alert numbers. Be concise.

SOURCES:
{context[:1800]}

QUESTION: {query}

ANSWER:"""
    try:
        r = http_requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.1, "num_predict": 350, "num_ctx": 2048}},
            timeout=TIMEOUT
        )
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"LLM unavailable: {e}"

# ── API Routes ────────────────────────────────────────────────────────────────
@app.post("/query")
async def query(req: QueryRequest):
    # DELTA fusion
    fused_vec, lang, en_query = delta_fuse(req.query)

    # Retrieve
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=fused_vec,
        limit=req.top_k,
        with_payload=True,
    ).points

    alert_numbers = [r.payload.get("alert_number","") for r in results]

    # Hyper-RAG verification
    verification = get_verification(alert_numbers, hyperedge_idx)

    # Build context
    context_parts = []
    for r in results:
        p = r.payload
        context_parts.append(
            f"[Alert {p.get('alert_number','?')} | {p.get('drug_name','?')} | "
            f"{p.get('date','?')} | Countries: {', '.join(p.get('countries',[]))}]\n"
            f"{p.get('text','')[:300]}"
        )
    context = "\n\n".join(context_parts)

    # LLM answer
    answer = ask_ollama(req.query, context)

    return {
        "query":        req.query,
        "language":     lang,
        "en_query":     en_query,
        "results": [{
            "alert_number": r.payload.get("alert_number",""),
            "drug_name":    r.payload.get("drug_name",""),
            "alert_type":   r.payload.get("alert_type",""),
            "date":         r.payload.get("date",""),
            "countries":    r.payload.get("countries",[]),
            "who_regions":  r.payload.get("who_regions",[]),
            "source_url":   r.payload.get("source_url",""),
            "score":        round(r.score, 4),
            "text":         r.payload.get("text","")[:400],
        } for r in results],
        "verification": verification[:8],
        "answer":       answer,
    }

@app.get("/timeline/{drug}")
async def timeline(drug: str):
    drug_lower = drug.lower()
    matches = {}
    for key, events in temporal_graph["drug_timeline"].items():
        if drug_lower in key:
            matches[key] = events
    return {"drug": drug, "timelines": matches}

@app.get("/stats")
async def stats():
    return {
        "total_alerts":    len(all_alerts),
        "total_vectors":   qdrant.count(COLLECTION_NAME).count,
        "total_hyperedges": len(hyperedge_idx["hyperedges"]),
        "country_counts":  dict(list(temporal_graph["country_counts"].items())[:15]),
        "yearly_counts":   temporal_graph["yearly_counts"],
        "alert_types": {
            "falsified":   sum(1 for a in all_alerts if a.get("alert_type") == "falsified"),
            "substandard": sum(1 for a in all_alerts if a.get("alert_type") == "substandard"),
            "both":        sum(1 for a in all_alerts if a.get("alert_type") == "both"),
        }
    }

@app.get("/alerts")
async def get_alerts():
    return {"alerts": all_alerts}

# ── Dashboard HTML ────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Counterfeit Drug Intelligence Network</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:       #0a0a0a;
    --surface:  #111111;
    --border:   #1e1e1e;
    --border2:  #2a2a2a;
    --text:     #e8e8e8;
    --muted:    #666;
    --accent:   #c8f542;
    --red:      #ff4444;
    --amber:    #ffaa00;
    --blue:     #4488ff;
    --mono:     'IBM Plex Mono', monospace;
    --sans:     'IBM Plex Sans', sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    line-height: 1.6;
    min-height: 100vh;
  }

  /* Layout */
  .shell { display: grid; grid-template-columns: 260px 1fr; min-height: 100vh; }
  .sidebar {
    background: var(--surface);
    border-right: 1px solid var(--border);
    padding: 24px 0;
    position: sticky;
    top: 0;
    height: 100vh;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
  }
  .main { padding: 32px; overflow-y: auto; }

  /* Sidebar */
  .logo {
    padding: 0 20px 24px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
  }
  .logo-tag {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--accent);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 4px;
  }
  .logo-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--text);
    line-height: 1.3;
  }
  .nav-section {
    padding: 0 20px;
    margin-bottom: 24px;
  }
  .nav-label {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
  }
  .nav-item {
    display: block;
    padding: 8px 12px;
    border-radius: 4px;
    color: var(--muted);
    cursor: pointer;
    font-size: 13px;
    border: none;
    background: none;
    width: 100%;
    text-align: left;
    transition: all 0.15s;
    margin-bottom: 2px;
  }
  .nav-item:hover { background: var(--border); color: var(--text); }
  .nav-item.active { background: var(--border2); color: var(--accent); }

  .stat-block {
    padding: 0 20px;
    margin-top: auto;
    border-top: 1px solid var(--border);
    padding-top: 20px;
  }
  .stat-row { display: flex; justify-content: space-between; padding: 4px 0; }
  .stat-key { font-family: var(--mono); font-size: 11px; color: var(--muted); }
  .stat-val { font-family: var(--mono); font-size: 11px; color: var(--accent); }

  /* Header */
  .page-header {
    display: flex;
    align-items: baseline;
    gap: 16px;
    margin-bottom: 28px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
  }
  .page-title { font-size: 20px; font-weight: 600; }
  .page-sub { font-size: 12px; color: var(--muted); font-family: var(--mono); }

  /* Search */
  .search-bar {
    display: flex;
    gap: 10px;
    margin-bottom: 28px;
  }
  .search-input {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: 6px;
    padding: 12px 16px;
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    outline: none;
    transition: border-color 0.2s;
  }
  .search-input:focus { border-color: var(--accent); }
  .search-input::placeholder { color: var(--muted); }
  .search-btn {
    background: var(--accent);
    color: #000;
    border: none;
    border-radius: 6px;
    padding: 12px 24px;
    font-family: var(--mono);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
    cursor: pointer;
    transition: opacity 0.2s;
  }
  .search-btn:hover { opacity: 0.85; }
  .search-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  /* Options */
  .options-row {
    display: flex;
    gap: 16px;
    margin-bottom: 24px;
    flex-wrap: wrap;
  }
  .toggle-label {
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    user-select: none;
  }
  .toggle-label input { accent-color: var(--accent); }
  .toggle-label:hover { color: var(--text); }

  /* Status bar */
  .status-bar {
    display: flex;
    gap: 20px;
    padding: 10px 16px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 24px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
  }
  .status-item { display: flex; gap: 6px; }
  .status-key { }
  .status-val { color: var(--text); }
  .status-val.ok { color: var(--accent); }
  .status-val.warn { color: var(--amber); }

  /* Grid */
  .results-grid {
    display: grid;
    grid-template-columns: 1fr 340px;
    gap: 20px;
    margin-bottom: 28px;
  }

  /* Alert cards */
  .alert-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
    cursor: default;
  }
  .alert-card:hover { border-color: var(--border2); }
  .card-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px; }
  .card-alert-num {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--accent);
    letter-spacing: 1px;
  }
  .card-score {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    background: var(--border);
    padding: 2px 8px;
    border-radius: 3px;
  }
  .card-drug { font-size: 14px; font-weight: 500; color: var(--text); margin-bottom: 6px; }
  .card-meta {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 10px;
  }
  .card-tag {
    font-family: var(--mono);
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 3px;
    border: 1px solid;
  }
  .tag-falsified { color: var(--red); border-color: #441111; background: #1a0808; }
  .tag-substandard { color: var(--amber); border-color: #443311; background: #1a1000; }
  .tag-both { color: #ff8844; border-color: #442211; background: #1a0c00; }
  .tag-country { color: var(--blue); border-color: #112244; background: #080e1a; }
  .card-text { font-size: 12px; color: var(--muted); line-height: 1.6; }
  .card-link { font-family: var(--mono); font-size: 10px; color: var(--blue); text-decoration: none; margin-top: 8px; display: inline-block; }
  .card-link:hover { color: var(--accent); }

  /* Verification panel */
  .verify-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px;
    position: sticky;
    top: 32px;
    max-height: 80vh;
    overflow-y: auto;
  }
  .panel-title {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
  }
  .verify-item {
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 10px;
  }
  .verify-item:last-child { border-bottom: none; }
  .verify-entity { font-size: 12px; color: var(--text); flex: 1; }
  .verify-badge {
    font-family: var(--mono);
    font-size: 9px;
    padding: 2px 6px;
    border-radius: 3px;
    white-space: nowrap;
  }
  .badge-high { background: #0f2010; color: var(--accent); border: 1px solid #1e4020; }
  .badge-low  { background: #201000; color: var(--amber); border: 1px solid #402000; }
  .verify-sources { font-family: var(--mono); font-size: 10px; color: var(--muted); margin-top: 2px; }

  /* Answer block */
  .answer-block {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 6px;
    padding: 20px;
    margin-bottom: 24px;
  }
  .answer-label {
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 10px;
  }
  .answer-text { font-size: 13px; color: var(--text); line-height: 1.8; white-space: pre-wrap; }

  /* Timeline */
  .timeline-section { margin-top: 32px; }
  .timeline-input-row { display: flex; gap: 10px; margin-bottom: 20px; }
  .timeline-events { position: relative; padding-left: 24px; }
  .timeline-line {
    position: absolute;
    left: 7px;
    top: 0;
    bottom: 0;
    width: 1px;
    background: var(--border2);
  }
  .timeline-event {
    position: relative;
    padding: 12px 0 12px 20px;
    border-bottom: 1px solid var(--border);
  }
  .timeline-event:last-child { border-bottom: none; }
  .timeline-dot {
    position: absolute;
    left: -17px;
    top: 18px;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--accent);
  }
  .timeline-date { font-family: var(--mono); font-size: 11px; color: var(--muted); }
  .timeline-drug { font-size: 13px; font-weight: 500; color: var(--text); margin: 2px 0; }
  .timeline-countries { font-size: 11px; color: var(--blue); }

  /* Stats view */
  .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 28px; }
  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 20px;
  }
  .stat-card-label { font-family: var(--mono); font-size: 10px; color: var(--muted); letter-spacing: 1px; text-transform: uppercase; margin-bottom: 8px; }
  .stat-card-value { font-size: 32px; font-weight: 600; font-family: var(--mono); color: var(--accent); }
  .country-bar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
  .country-name { font-size: 12px; color: var(--text); width: 120px; flex-shrink: 0; }
  .country-bar-bg { flex: 1; height: 4px; background: var(--border); border-radius: 2px; }
  .country-bar-fill { height: 100%; background: var(--accent); border-radius: 2px; transition: width 0.4s; }
  .country-count { font-family: var(--mono); font-size: 11px; color: var(--muted); width: 20px; text-align: right; }

  /* Loading */
  .loading { color: var(--muted); font-family: var(--mono); font-size: 12px; padding: 40px 0; text-align: center; }
  .dot-anim::after { content: '...'; animation: dots 1.2s steps(4, end) infinite; }
  @keyframes dots { 0%,20%{content:'.'} 40%{content:'..'} 60%,100%{content:'...'} }

  /* Section separator */
  .section-sep {
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 14px;
    margin-top: 24px;
  }

  /* Alert list table */
  .alert-table { width: 100%; border-collapse: collapse; }
  .alert-table th {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 1px;
    color: var(--muted);
    text-align: left;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    text-transform: uppercase;
  }
  .alert-table td {
    font-size: 12px;
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
    vertical-align: top;
  }
  .alert-table tr:hover td { background: var(--surface); }
  .alert-table .mono { font-family: var(--mono); font-size: 11px; }
  .hidden { display: none; }
</style>
</head>
<body>
<div class="shell">

<!-- Sidebar -->
<nav class="sidebar">
  <div class="logo">
    <div class="logo-tag">WHO / INTERPOL</div>
    <div class="logo-title">Counterfeit Drug<br>Intelligence Network</div>
  </div>

  <div class="nav-section">
    <div class="nav-label">Navigation</div>
    <button class="nav-item active" onclick="showView('search')">Search Intelligence</button>
    <button class="nav-item" onclick="showView('timeline')">Drug Timeline</button>
    <button class="nav-item" onclick="showView('alerts')">Alert Database</button>
    <button class="nav-item" onclick="showView('stats')">Statistics</button>
  </div>

  <div class="stat-block" id="sidebar-stats">
    <div class="nav-label" style="margin-bottom:10px">System</div>
    <div class="stat-row"><span class="stat-key">alerts</span><span class="stat-val" id="ss-alerts">--</span></div>
    <div class="stat-row"><span class="stat-key">vectors</span><span class="stat-val" id="ss-vectors">--</span></div>
    <div class="stat-row"><span class="stat-key">hyperedges</span><span class="stat-val" id="ss-hyperedges">--</span></div>
    <div class="stat-row"><span class="stat-key">layers</span><span class="stat-val ok">4 active</span></div>
  </div>
</nav>

<!-- Main -->
<main class="main">

  <!-- SEARCH VIEW -->
  <div id="view-search">
    <div class="page-header">
      <span class="page-title">Intelligence Search</span>
      <span class="page-sub">RAG + T-GRAG + Hyper-RAG + DELTA</span>
    </div>

    <div class="search-bar">
      <input class="search-input" id="query-input" type="text"
             placeholder="Search in any language — English, Urdu, French, Arabic..."
             onkeydown="if(event.key==='Enter') runQuery()">
      <button class="search-btn" id="search-btn" onclick="runQuery()">SEARCH</button>
    </div>

    <div class="options-row">
      <label class="toggle-label"><input type="checkbox" id="opt-temporal" checked> T-GRAG temporal</label>
      <label class="toggle-label"><input type="checkbox" id="opt-hyper" checked> Hyper-RAG verify</label>
      <label class="toggle-label"><input type="checkbox" id="opt-delta" checked> DELTA multilingual</label>
      <label class="toggle-label">
        Top-K:
        <select id="opt-topk" style="background:var(--surface);border:1px solid var(--border2);color:var(--text);border-radius:3px;padding:2px 6px;font-family:var(--mono);font-size:11px;">
          <option value="3">3</option>
          <option value="5" selected>5</option>
          <option value="8">8</option>
        </select>
      </label>
    </div>

    <div id="status-bar" class="status-bar hidden">
      <div class="status-item"><span class="status-key">lang:</span><span class="status-val" id="st-lang">--</span></div>
      <div class="status-item"><span class="status-key">en_query:</span><span class="status-val" id="st-enq">--</span></div>
      <div class="status-item"><span class="status-key">results:</span><span class="status-val" id="st-results">--</span></div>
      <div class="status-item"><span class="status-key">verified:</span><span class="status-val ok" id="st-verified">--</span></div>
    </div>

    <div id="loading" class="hidden"><div class="loading"><span class="dot-anim">Querying intelligence</span></div></div>

    <div id="answer-section" class="hidden">
      <div class="answer-block">
        <div class="answer-label">LLM Answer — Grounded by Hyper-RAG</div>
        <div class="answer-text" id="answer-text"></div>
      </div>
    </div>

    <div id="results-section" class="hidden">
      <div class="section-sep">Retrieved Alerts</div>
      <div class="results-grid">
        <div id="alert-cards"></div>
        <div class="verify-panel">
          <div class="panel-title">Hyper-RAG Verification</div>
          <div id="verify-list"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- TIMELINE VIEW -->
  <div id="view-timeline" class="hidden">
    <div class="page-header">
      <span class="page-title">Drug Alert Timeline</span>
      <span class="page-sub">T-GRAG temporal graph traversal</span>
    </div>
    <div class="timeline-input-row">
      <input class="search-input" id="timeline-input" type="text"
             placeholder="Enter drug name — e.g. coartem, ozempic, covishield"
             onkeydown="if(event.key==='Enter') runTimeline()">
      <button class="search-btn" onclick="runTimeline()">TRACE</button>
    </div>
    <div id="timeline-results"></div>
  </div>

  <!-- ALERTS VIEW -->
  <div id="view-alerts" class="hidden">
    <div class="page-header">
      <span class="page-title">Alert Database</span>
      <span class="page-sub">80 WHO Medical Product Alerts (2013 - 2026)</span>
    </div>
    <div id="alerts-table-container"><div class="loading">Loading alerts...</div></div>
  </div>

  <!-- STATS VIEW -->
  <div id="view-stats" class="hidden">
    <div class="page-header">
      <span class="page-title">System Statistics</span>
      <span class="page-sub">Knowledge graph and corpus analytics</span>
    </div>
    <div id="stats-container"><div class="loading">Loading stats...</div></div>
  </div>

</main>
</div>

<script>
const API = 'http://localhost:8000';

// ── View switching ────────────────────────────────────────────────────────────
function showView(name) {
  ['search','timeline','alerts','stats'].forEach(v => {
    document.getElementById(`view-${v}`).classList.toggle('hidden', v !== name);
  });
  document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  if (name === 'stats') loadStats();
  if (name === 'alerts') loadAlerts();
}

// ── Sidebar stats ─────────────────────────────────────────────────────────────
async function loadSidebarStats() {
  try {
    const r = await fetch(`${API}/stats`);
    const d = await r.json();
    document.getElementById('ss-alerts').textContent     = d.total_alerts;
    document.getElementById('ss-vectors').textContent    = d.total_vectors;
    document.getElementById('ss-hyperedges').textContent = d.total_hyperedges;
  } catch(e) {}
}
loadSidebarStats();

// ── Search ────────────────────────────────────────────────────────────────────
async function runQuery() {
  const query = document.getElementById('query-input').value.trim();
  if (!query) return;

  document.getElementById('search-btn').disabled = true;
  document.getElementById('loading').classList.remove('hidden');
  document.getElementById('answer-section').classList.add('hidden');
  document.getElementById('results-section').classList.add('hidden');
  document.getElementById('status-bar').classList.add('hidden');

  try {
    const r = await fetch(`${API}/query`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        query,
        top_k: parseInt(document.getElementById('opt-topk').value),
        use_temporal: document.getElementById('opt-temporal').checked,
        use_hyper:    document.getElementById('opt-hyper').checked,
        use_delta:    document.getElementById('opt-delta').checked,
      })
    });
    const d = await r.json();
    renderResults(d);
  } catch(e) {
    document.getElementById('loading').innerHTML = `<div class="loading">Backend not reachable. Make sure dashboard_backend.py is running.</div>`;
  } finally {
    document.getElementById('search-btn').disabled = false;
  }
}

function renderResults(d) {
  document.getElementById('loading').classList.add('hidden');

  // Status bar
  document.getElementById('st-lang').textContent    = d.language || 'en';
  document.getElementById('st-enq').textContent     = (d.en_query || '').substring(0,50);
  document.getElementById('st-results').textContent = d.results.length;
  document.getElementById('st-verified').textContent = d.verification.filter(v=>v.confidence==='HIGH').length + ' HIGH';
  document.getElementById('status-bar').classList.remove('hidden');

  // Answer
  if (d.answer) {
    document.getElementById('answer-text').textContent = d.answer;
    document.getElementById('answer-section').classList.remove('hidden');
  }

  // Alert cards
  const cards = document.getElementById('alert-cards');
  cards.innerHTML = d.results.map(r => {
    const typeClass = r.alert_type === 'falsified' ? 'tag-falsified' :
                      r.alert_type === 'substandard' ? 'tag-substandard' : 'tag-both';
    const countries = r.countries.slice(0,4).map(c =>
      `<span class="card-tag tag-country">${c}</span>`).join('');
    return `
    <div class="alert-card">
      <div class="card-header">
        <span class="card-alert-num">ALERT ${r.alert_number}</span>
        <span class="card-score">${r.score}</span>
      </div>
      <div class="card-drug">${r.drug_name || 'Unknown product'}</div>
      <div class="card-meta">
        ${r.alert_type ? `<span class="card-tag ${typeClass}">${r.alert_type}</span>` : ''}
        ${countries}
        <span class="card-tag" style="color:var(--muted);border-color:var(--border)">${r.date}</span>
      </div>
      <div class="card-text">${r.text.substring(0,200)}...</div>
      ${r.source_url ? `<a class="card-link" href="${r.source_url}" target="_blank">WHO source</a>` : ''}
    </div>`;
  }).join('');

  // Verification panel
  const vlist = document.getElementById('verify-list');
  if (d.verification.length === 0) {
    vlist.innerHTML = '<div style="color:var(--muted);font-size:12px;padding:10px 0">No cross-verified facts found</div>';
  } else {
    vlist.innerHTML = d.verification.map(v => {
      const entity = v.entity
        .replace('DRUG:','Drug: ').replace('COUNTRY:','Country: ')
        .replace('KEYWORD:','Topic: ').replace('TYPE:','Type: ').replace('YEAR:','Year: ');
      const badgeClass = v.confidence === 'HIGH' ? 'badge-high' : 'badge-low';
      return `
      <div class="verify-item">
        <div>
          <div class="verify-entity">${entity}</div>
          <div class="verify-sources">${v.confirmed_by.slice(0,3).join(', ')}</div>
        </div>
        <span class="verify-badge ${badgeClass}">${v.source_count} src</span>
      </div>`;
    }).join('');
  }

  document.getElementById('results-section').classList.remove('hidden');
}

// ── Timeline ──────────────────────────────────────────────────────────────────
async function runTimeline() {
  const drug = document.getElementById('timeline-input').value.trim();
  if (!drug) return;
  const container = document.getElementById('timeline-results');
  container.innerHTML = '<div class="loading"><span class="dot-anim">Tracing timeline</span></div>';
  try {
    const r = await fetch(`${API}/timeline/${encodeURIComponent(drug)}`);
    const d = await r.json();
    const timelines = d.timelines;
    const keys = Object.keys(timelines);
    if (keys.length === 0) {
      container.innerHTML = '<div class="loading">No alerts found for this drug</div>';
      return;
    }
    let html = '';
    for (const key of keys) {
      const events = timelines[key];
      html += `<div class="section-sep">${key}</div>`;
      html += '<div class="timeline-events"><div class="timeline-line"></div>';
      html += events.map(e => `
        <div class="timeline-event">
          <div class="timeline-dot"></div>
          <div class="timeline-date">${e.date}</div>
          <div class="timeline-drug">Alert ${e.alert_number} — ${e.alert_type || 'alert'}</div>
          <div class="timeline-countries">${e.countries.join(', ')}</div>
        </div>`).join('');
      html += '</div>';
    }
    container.innerHTML = html;
  } catch(e) {
    container.innerHTML = '<div class="loading">Error fetching timeline</div>';
  }
}

// ── Alert database ────────────────────────────────────────────────────────────
async function loadAlerts() {
  const container = document.getElementById('alerts-table-container');
  try {
    const r = await fetch(`${API}/alerts`);
    const d = await r.json();
    const alerts = d.alerts;
    let html = `<table class="alert-table">
      <thead><tr>
        <th>Alert</th><th>Drug</th><th>Type</th><th>Date</th><th>Countries</th>
      </tr></thead><tbody>`;
    html += alerts.map(a => {
      const typeClass = a.alert_type === 'falsified' ? 'tag-falsified' :
                        a.alert_type === 'substandard' ? 'tag-substandard' :
                        a.alert_type === 'both' ? 'tag-both' : '';
      return `<tr>
        <td class="mono">${a.alert_number || '--'}</td>
        <td>${a.drug_name || '--'}</td>
        <td>${a.alert_type ? `<span class="card-tag ${typeClass}">${a.alert_type}</span>` : '--'}</td>
        <td class="mono" style="font-size:11px">${a.date || '--'}</td>
        <td style="color:var(--blue);font-size:11px">${(a.countries||[]).join(', ') || '--'}</td>
      </tr>`;
    }).join('');
    html += '</tbody></table>';
    container.innerHTML = html;
  } catch(e) {
    container.innerHTML = '<div class="loading">Error loading alerts</div>';
  }
}

// ── Stats ─────────────────────────────────────────────────────────────────────
async function loadStats() {
  const container = document.getElementById('stats-container');
  try {
    const r = await fetch(`${API}/stats`);
    const d = await r.json();
    const maxCountry = Math.max(...Object.values(d.country_counts));
    const countryBars = Object.entries(d.country_counts).slice(0,12).map(([country,count]) => `
      <div class="country-bar-row">
        <span class="country-name">${country}</span>
        <div class="country-bar-bg"><div class="country-bar-fill" style="width:${Math.round(count/maxCountry*100)}%"></div></div>
        <span class="country-count">${count}</span>
      </div>`).join('');

    container.innerHTML = `
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-card-label">Total Alerts</div>
          <div class="stat-card-value">${d.total_alerts}</div>
        </div>
        <div class="stat-card">
          <div class="stat-card-label">Vector Index</div>
          <div class="stat-card-value">${d.total_vectors}</div>
        </div>
        <div class="stat-card">
          <div class="stat-card-label">Hyperedges</div>
          <div class="stat-card-value">${d.total_hyperedges}</div>
        </div>
        <div class="stat-card">
          <div class="stat-card-label">Falsified</div>
          <div class="stat-card-value" style="color:var(--red)">${d.alert_types.falsified}</div>
        </div>
        <div class="stat-card">
          <div class="stat-card-label">Substandard</div>
          <div class="stat-card-value" style="color:var(--amber)">${d.alert_types.substandard}</div>
        </div>
        <div class="stat-card">
          <div class="stat-card-label">Both Types</div>
          <div class="stat-card-value" style="color:#ff8844">${d.alert_types.both}</div>
        </div>
      </div>
      <div class="section-sep">Most Affected Countries</div>
      ${countryBars}
    `;
  } catch(e) {
    container.innerHTML = '<div class="loading">Error loading stats</div>';
  }
}
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
