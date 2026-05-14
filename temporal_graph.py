"""
Temporal Knowledge Graph (T-GRAG Layer)
========================================
Based on: T-GRAG arxiv 2508.01680

Builds a time-stamped knowledge graph over WHO alerts where:
- Nodes = drugs, countries, manufacturers, alert types
- Edges = relationships with timestamps
- Temporal queries = "what happened with drug X between year A and B?"

This implements the core T-GRAG concept:
1. Temporal Knowledge Graph Generator
2. Temporal Query Decomposition
3. Graph-aware retrieval combining vector search + graph traversal

Install:
    pip install qdrant-client sentence-transformers requests networkx
    python temporal_graph.py
"""

import os, re, json, uuid
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import networkx as nx
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import requests

# ── Config ────────────────────────────────────────────────────────────────────
ALERTS_DIR      = "who_alerts_v2"
GRAPH_FILE      = "temporal_graph.json"
COLLECTION_NAME = "counterfeit_drug_alerts"
EMBED_MODEL     = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_MODEL    = "tinyllama"
TIMEOUT         = 300

# ── Parse alert file ──────────────────────────────────────────────────────────
def parse_alert_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    def field(name):
        m = re.search(rf"^{name}:\s*(.+)$", content, re.MULTILINE)
        return m.group(1).strip() if m else ""

    parts = content.split("=" * 60)
    body  = parts[1].strip() if len(parts) > 1 else content

    # Parse year from date string
    date_str = field("DATE")
    year = None
    m = re.search(r"(\d{4})", date_str)
    if m:
        year = int(m.group(1))

    # Parse month
    month = None
    months = {"january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
              "july":7,"august":8,"september":9,"october":10,"november":11,"december":12}
    for month_name, month_num in months.items():
        if month_name in date_str.lower():
            month = month_num
            break

    return {
        "alert_number": field("ALERT NUMBER"),
        "drug_name":    field("DRUG NAME"),
        "alert_type":   field("ALERT TYPE"),
        "date_str":     date_str,
        "year":         year,
        "month":        month,
        "countries":    [c.strip() for c in field("COUNTRIES").split(",") if c.strip()],
        "who_regions":  [r.strip() for r in field("WHO REGIONS").split(",") if r.strip()],
        "source_url":   field("SOURCE URL"),
        "title":        field("TITLE"),
        "body":         body,
        "filename":     os.path.basename(filepath),
    }

# ────────────────────────────────────────────────────────────────────────────
# STEP 1: Build Temporal Knowledge Graph
# T-GRAG Paper: "Temporal Knowledge Graph Generator that creates
# time-stamped, evolving graph structures"
# ────────────────────────────────────────────────────────────────────────────

def build_temporal_graph(alerts):
    """
    Build a directed temporal knowledge graph.
    
    Node types:
        DRUG      — e.g. "OZEMPIC (semaglutide)"
        COUNTRY   — e.g. "Pakistan"
        REGION    — e.g. "AFRO"
        ALERT     — e.g. "2/2024"
        TYPE      — "falsified" | "substandard" | "both"
    
    Edge types (all timestamped):
        ALERT -> DRUG      (alert concerns drug)
        ALERT -> COUNTRY   (alert detected in country)
        ALERT -> REGION    (alert affects region)
        ALERT -> TYPE      (alert is of type)
        DRUG  -> COUNTRY   (drug found in country, at time T)
        DRUG  -> DRUG      (UPDATE relationship — same drug, later alert)
    """
    G = nx.DiGraph()

    # Sort alerts by year then month for temporal ordering
    sorted_alerts = sorted(
        alerts,
        key=lambda a: (a.get("year") or 0, a.get("month") or 0)
    )

    # Track drugs we've seen — for UPDATE edges
    drug_to_alerts = defaultdict(list)

    for alert in sorted_alerts:
        alert_id  = f"ALERT:{alert['alert_number']}"
        drug_name = alert["drug_name"].strip().lower() if alert["drug_name"] else ""
        year      = alert["year"]
        month     = alert["month"]
        timestamp = f"{year}-{month:02d}" if year and month else str(year or "unknown")

        # Add ALERT node
        G.add_node(alert_id, 
                   type="ALERT",
                   alert_number=alert["alert_number"],
                   drug_name=alert["drug_name"],
                   alert_type=alert["alert_type"],
                   date=alert["date_str"],
                   year=year,
                   month=month,
                   timestamp=timestamp,
                   source_url=alert["source_url"],
                   body=alert["body"][:500])

        # Add DRUG node and edge
        if drug_name and drug_name != "unknown":
            drug_id = f"DRUG:{drug_name}"
            if not G.has_node(drug_id):
                G.add_node(drug_id, type="DRUG", name=alert["drug_name"],
                           first_seen=year, last_seen=year)
            else:
                G.nodes[drug_id]["last_seen"] = year

            G.add_edge(alert_id, drug_id,
                      relation="CONCERNS",
                      timestamp=timestamp,
                      year=year)

            # Track for UPDATE edges
            drug_to_alerts[drug_name].append(alert_id)

        # Add COUNTRY nodes and edges
        for country in alert["countries"]:
            country_id = f"COUNTRY:{country}"
            if not G.has_node(country_id):
                G.add_node(country_id, type="COUNTRY", name=country,
                           alert_count=0)
            G.nodes[country_id]["alert_count"] = G.nodes[country_id].get("alert_count", 0) + 1

            G.add_edge(alert_id, country_id,
                      relation="DETECTED_IN",
                      timestamp=timestamp,
                      year=year)

            # Drug -> Country edge (temporal)
            if drug_name and drug_name != "unknown":
                G.add_edge(f"DRUG:{drug_name}", country_id,
                          relation="FOUND_IN",
                          timestamp=timestamp,
                          year=year,
                          alert=alert["alert_number"])

        # Add REGION nodes and edges
        for region in alert["who_regions"]:
            region_id = f"REGION:{region}"
            if not G.has_node(region_id):
                G.add_node(region_id, type="REGION", name=region)
            G.add_edge(alert_id, region_id,
                      relation="AFFECTS_REGION",
                      timestamp=timestamp,
                      year=year)

        # Add TYPE node and edge
        if alert["alert_type"]:
            type_id = f"TYPE:{alert['alert_type']}"
            if not G.has_node(type_id):
                G.add_node(type_id, type="TYPE", name=alert["alert_type"])
            G.add_edge(alert_id, type_id,
                      relation="IS_TYPE",
                      timestamp=timestamp,
                      year=year)

    # Add UPDATE edges between alerts about the same drug (T-GRAG temporal linking)
    for drug_name, alert_ids in drug_to_alerts.items():
        if len(alert_ids) > 1:
            for i in range(len(alert_ids) - 1):
                G.add_edge(alert_ids[i], alert_ids[i+1],
                          relation="UPDATED_BY",
                          timestamp="temporal_link")

    return G

# ────────────────────────────────────────────────────────────────────────────
# STEP 2: Temporal Query Decomposition
# T-GRAG Paper: "Temporal Query Decomposition mechanism that breaks
# complex temporal queries into manageable sub-queries"
# ────────────────────────────────────────────────────────────────────────────

def decompose_temporal_query(query):
    """
    Detect if query has temporal intent and decompose it.
    Returns: {type, drug, country, year_from, year_to, sub_queries}
    """
    query_lower = query.lower()
    
    # Detect year range
    years = re.findall(r"\b(20\d{2})\b", query)
    year_from = int(min(years)) if years else None
    year_to   = int(max(years)) if years else None

    # Detect temporal keywords
    temporal_keywords = ["between", "from", "since", "until", "before", "after",
                         "evolution", "history", "trend", "over time", "changed",
                         "latest", "recent", "first", "earliest"]
    is_temporal = any(kw in query_lower for kw in temporal_keywords) or len(years) > 0

    # Detect drug mention
    drug_keywords = []
    common_drugs  = ["coartem", "ozempic", "semaglutide", "avastin", "covishield",
                     "astrazeneca", "remdesivir", "chloroquine", "amoxicillin",
                     "dextromethorphan", "imfinzi", "simulect", "ibrance",
                     "oxycontin", "fentanyl", "diazepam", "quinine", "augmentin"]
    for drug in common_drugs:
        if drug in query_lower:
            drug_keywords.append(drug)

    # Detect country mention
    countries = ["pakistan", "india", "nigeria", "cameroon", "niger", "chad",
                 "uganda", "kenya", "ghana", "iran", "brazil", "usa", "uk",
                 "indonesia", "philippines", "myanmar", "egypt", "turkey"]
    found_countries = [c for c in countries if c in query_lower]

    # Build sub-queries for temporal decomposition
    sub_queries = [query]  # always include original
    if is_temporal and year_from and year_to and year_from != year_to:
        # Decompose into per-year sub-queries
        for year in range(year_from, year_to + 1):
            sub_queries.append(f"{' '.join(drug_keywords or ['medicine'])} alert {year}")

    return {
        "original":     query,
        "is_temporal":  is_temporal,
        "drug":         drug_keywords[0] if drug_keywords else None,
        "countries":    found_countries,
        "year_from":    year_from,
        "year_to":      year_to,
        "sub_queries":  sub_queries,
    }

# ────────────────────────────────────────────────────────────────────────────
# STEP 3: Graph-aware Retrieval
# T-GRAG Paper: "Three-layer Interactive Retriever that progressively
# filters and refines retrieval across temporal subgraphs"
# ────────────────────────────────────────────────────────────────────────────

def graph_retrieve(query_info, G, embedder, qdrant_client, top_k=5):
    """
    Layer 1: Vector search for relevant alert IDs
    Layer 2: Graph traversal to find connected nodes
    Layer 3: Temporal filtering by year range
    """
    results = []

    # Layer 1 — Vector search for each sub-query
    seen_alerts = set()
    for sub_q in query_info["sub_queries"]:
        qvec = embedder.encode(sub_q).tolist()
        hits = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=qvec,
            limit=top_k,
            with_payload=True,
        ).points
        for hit in hits:
            alert_num = hit.payload.get("alert_number", "")
            if alert_num not in seen_alerts:
                seen_alerts.add(alert_num)
                results.append({
                    "alert_number": alert_num,
                    "score":        hit.score,
                    "payload":      hit.payload,
                    "source":       "vector",
                })

    # Layer 2 — Graph traversal
    # If drug mentioned, find all alerts about that drug in graph
    if query_info["drug"]:
        drug_node = f"DRUG:{query_info['drug']}"
        # Find similar drug nodes (case-insensitive partial match)
        for node_id in G.nodes():
            if G.nodes[node_id].get("type") == "DRUG":
                node_name = G.nodes[node_id].get("name", "").lower()
                if query_info["drug"] in node_name:
                    # Get all alerts connected to this drug
                    predecessors = list(G.predecessors(node_id))
                    for pred in predecessors:
                        if G.nodes[pred].get("type") == "ALERT":
                            alert_num = G.nodes[pred].get("alert_number", "")
                            if alert_num not in seen_alerts:
                                seen_alerts.add(alert_num)
                                results.append({
                                    "alert_number": alert_num,
                                    "score":        0.9,  # high score — direct graph match
                                    "payload":      G.nodes[pred],
                                    "source":       "graph",
                                })

    # Layer 3 — Temporal filtering
    if query_info["year_from"] or query_info["year_to"]:
        filtered = []
        for r in results:
            alert_node_id = f"ALERT:{r['alert_number']}"
            if G.has_node(alert_node_id):
                node_year = G.nodes[alert_node_id].get("year")
                if node_year:
                    in_range = True
                    if query_info["year_from"] and node_year < query_info["year_from"]:
                        in_range = False
                    if query_info["year_to"] and node_year > query_info["year_to"]:
                        in_range = False
                    if in_range:
                        filtered.append(r)
                else:
                    filtered.append(r)  # keep if no year info
            else:
                filtered.append(r)
        results = filtered

    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

# ── Graph statistics ──────────────────────────────────────────────────────────
def print_graph_stats(G):
    node_types = defaultdict(int)
    edge_types = defaultdict(int)
    for _, data in G.nodes(data=True):
        node_types[data.get("type", "unknown")] += 1
    for _, _, data in G.edges(data=True):
        edge_types[data.get("relation", "unknown")] += 1

    print(f"\n📊 Temporal Knowledge Graph Stats:")
    print(f"   Total nodes : {G.number_of_nodes()}")
    print(f"   Total edges : {G.number_of_edges()}")
    print(f"\n   Node types:")
    for t, count in sorted(node_types.items()):
        print(f"      {t:12} : {count}")
    print(f"\n   Edge types:")
    for t, count in sorted(edge_types.items()):
        print(f"      {t:20} : {count}")

def print_drug_timeline(drug_query, G):
    """Show timeline of alerts for a specific drug."""
    drug_query = drug_query.lower()
    print(f"\n📅 Timeline for '{drug_query}':")
    
    timeline = []
    for node_id, data in G.nodes(data=True):
        if data.get("type") == "ALERT":
            drug = data.get("drug_name", "").lower()
            if drug_query in drug:
                timeline.append((
                    data.get("year", 9999),
                    data.get("month", 99),
                    data.get("alert_number", "?"),
                    data.get("drug_name", "?"),
                    data.get("date", "?"),
                    data.get("alert_type", "?"),
                ))
    
    timeline.sort()
    if not timeline:
        print(f"   No alerts found for '{drug_query}'")
        return
    
    for year, month, alert_num, drug, date, atype in timeline:
        # Get countries from graph
        alert_node = f"ALERT:{alert_num}"
        countries  = []
        for successor in G.successors(alert_node):
            if G.nodes[successor].get("type") == "COUNTRY":
                countries.append(G.nodes[successor].get("name", ""))
        print(f"   {date:20} | Alert {alert_num:8} | {atype:12} | {', '.join(countries[:3])}")

# ── Ollama answer ─────────────────────────────────────────────────────────────
def ask_ollama_with_graph(query, results, graph_context):
    context_parts = []
    for r in results:
        p = r["payload"]
        context_parts.append(
            f"[Alert {p.get('alert_number','?')} | {p.get('drug_name','?')} | "
            f"{p.get('date_str', p.get('date','?'))} | "
            f"Source: {r['source']}]\n"
            f"{p.get('body', p.get('text',''))[:400]}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""You are a drug safety analyst. Answer using ONLY these WHO alert sources. Cite alert numbers.

GRAPH CONTEXT:
{graph_context}

ALERT SOURCES:
{context[:2000]}

QUESTION: {query}

ANSWER (cite alert numbers, mention dates and countries):"""

    print("   Generating answer", end="", flush=True)
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.1, "num_predict": 400, "num_ctx": 2048}},
            timeout=TIMEOUT
        )
        print(" ✓")
        return r.json().get("response", "No response.").strip()
    except Exception as e:
        print()
        return f"❌ Ollama error: {e}"

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("T-GRAG: Temporal Knowledge Graph RAG")
    print("Based on arxiv 2508.01680")
    print("="*60)

    # Load all alerts
    print("\n[1/4] Loading alerts...")
    txt_files = sorted(Path(ALERTS_DIR).glob("*.txt"))
    alerts    = []
    for fp in txt_files:
        try:
            alerts.append(parse_alert_file(str(fp)))
        except Exception as e:
            print(f"   Error {fp.name}: {e}")
    print(f"   Loaded {len(alerts)} alerts")

    # Build temporal graph
    print("\n[2/4] Building Temporal Knowledge Graph...")
    G = build_temporal_graph(alerts)
    print_graph_stats(G)

    # Save graph to JSON for inspection
    graph_data = {
        "nodes": [{"id": n, **d} for n, d in G.nodes(data=True)],
        "edges": [{"from": u, "to": v, **d} for u, v, d in G.edges(data=True)],
    }
    with open(GRAPH_FILE, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    print(f"   Graph saved → {GRAPH_FILE}")

    # Load embedder and Qdrant
    print("\n[3/4] Loading embedder + Qdrant...")
    embedder = SentenceTransformer(EMBED_MODEL)
    client   = QdrantClient(path="qdrant_storage")
    print(f"   Qdrant vectors: {client.count(COLLECTION_NAME).count}")

    # Warm up Ollama
    print("\n[4/4] Warming up TinyLlama...")
    try:
        requests.post(OLLAMA_URL,
                     json={"model": OLLAMA_MODEL, "prompt": "Hi", "stream": False,
                           "options": {"num_predict": 3}},
                     timeout=60)
        print("   Warmed up ✓")
    except:
        print("   Ollama not responding — answers will show context only")

    # ── Demo temporal queries ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEMPORAL QUERY DEMOS")
    print("="*60)

    # Show drug timelines
    print_drug_timeline("coartem", G)
    print_drug_timeline("covishield", G)
    print_drug_timeline("coartem", G)

    temporal_queries = [
        "How did alerts about falsified medicines in Africa evolve between 2013 and 2017?",
        "What happened with COVID vaccine falsification alerts in 2021?",
        "Show me all alerts about contaminated paediatric medicines from 2022 to 2023",
        "Which countries had the most falsified medicine alerts over time?",
    ]

    for query in temporal_queries:
        print(f"\n{'='*60}")
        print(f"TEMPORAL QUERY: {query}")
        print(f"{'='*60}")

        # Decompose query
        query_info = decompose_temporal_query(query)
        print(f"\n🔍 Query Analysis:")
        print(f"   Temporal  : {query_info['is_temporal']}")
        print(f"   Drug      : {query_info['drug']}")
        print(f"   Year range: {query_info['year_from']} → {query_info['year_to']}")
        print(f"   Sub-queries: {len(query_info['sub_queries'])}")

        # Graph-aware retrieval
        results = graph_retrieve(query_info, G, embedder, client)
        print(f"\n📚 Retrieved {len(results)} alerts (vector + graph):")
        for r in results:
            p = r["payload"]
            countries = ", ".join(p.get("countries", []))[:35]
            print(f"   [{r['score']:.3f}|{r['source']:6}] "
                  f"Alert {p.get('alert_number','?'):8} | "
                  f"{p.get('drug_name','?')[:25]:25} | "
                  f"{p.get('date', p.get('date_str','?'))[:15]:15} | "
                  f"{countries}")

        # Build graph context string
        graph_context = f"Query covers years {query_info['year_from']} to {query_info['year_to']}. "
        graph_context += f"Found {len(results)} alerts via {'temporal graph + vector search' if query_info['is_temporal'] else 'vector search'}."

        answer = ask_ollama_with_graph(query, results, graph_context)
        print(f"\n🤖 Answer:")
        print("-"*60)
        print(answer)
        print("-"*60)

    # ── Interactive mode ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("INTERACTIVE MODE — try temporal queries like:")
    print("  'What happened with Coartem between 2013 and 2014?'")
    print("  'Show alerts from 2021 about COVID vaccines'")
    print("  'Which drugs were falsified in Pakistan over time?'")
    print("Type 'timeline <drug>' to see a drug's alert history")
    print("Type 'exit' to quit")
    print("="*60)

    while True:
        try:
            query = input("\n🔍 Question: ").strip()
        except KeyboardInterrupt:
            break
        if query.lower() in ["exit", "quit", "q", ""]:
            break

        if query.lower().startswith("timeline "):
            drug = query[9:].strip()
            print_drug_timeline(drug, G)
            continue

        query_info = decompose_temporal_query(query)
        results    = graph_retrieve(query_info, G, embedder, client)

        print(f"\n📚 Retrieved {len(results)} alerts:")
        for r in results:
            p = r["payload"]
            countries = ", ".join(p.get("countries", []))[:35]
            print(f"   [{r['score']:.3f}|{r['source']:6}] "
                  f"Alert {p.get('alert_number','?'):8} | "
                  f"{p.get('drug_name','?')[:25]:25} | "
                  f"{countries}")

        graph_context = (
            f"Temporal range: {query_info['year_from']} to {query_info['year_to']}. "
            f"Drug focus: {query_info['drug']}. "
            f"Retrieved via: {'graph+vector' if query_info['is_temporal'] else 'vector'}."
        )
        answer = ask_ollama_with_graph(query, results, graph_context)
        print(f"\n🤖 Answer:")
        print("-"*60)
        print(answer)
        print("-"*60)

    print("Done.")

if __name__ == "__main__":
    main()
