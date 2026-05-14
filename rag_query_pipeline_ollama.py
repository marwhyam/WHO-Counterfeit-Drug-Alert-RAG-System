"""
Counterfeit Drug Intelligence — RAG Query Pipeline (Ollama Version)
=====================================================================
Uses Ollama local LLM — no API key needed.

SETUP (do this first):
1. Download Ollama from https://ollama.com and install it
2. Open a terminal and run: ollama pull mistral
3. Keep ollama running (it runs in background automatically)
4. Then run this script: python rag_query_pipeline_ollama.py

Requirements:
    pip install qdrant-client sentence-transformers requests
"""

import os, re, uuid, json, requests
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ── Config ────────────────────────────────────────────────────────────────────
ALERTS_DIR      = "who_alerts_v2"
COLLECTION_NAME = "counterfeit_drug_alerts"
EMBED_MODEL     = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CHUNK_SIZE      = 300
CHUNK_OVERLAP   = 60
TOP_K           = 5
OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_MODEL    = "mistral"   # change to "tinyllama" if low RAM

# ── Parse alert file ──────────────────────────────────────────────────────────
def parse_alert_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    def field(name):
        m = re.search(rf"^{name}:\s*(.+)$", content, re.MULTILINE)
        return m.group(1).strip() if m else ""

    parts = content.split("=" * 60)
    body  = parts[1].strip() if len(parts) > 1 else content

    return {
        "alert_number": field("ALERT NUMBER"),
        "drug_name":    field("DRUG NAME"),
        "alert_type":   field("ALERT TYPE"),
        "date":         field("DATE"),
        "countries":    [c.strip() for c in field("COUNTRIES").split(",") if c.strip()],
        "who_regions":  [r.strip() for r in field("WHO REGIONS").split(",") if r.strip()],
        "source_url":   field("SOURCE URL"),
        "title":        field("TITLE"),
        "body":         body,
        "filename":     os.path.basename(filepath),
    }

# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += size - overlap
    return chunks if chunks else [text]

# ── Build Qdrant index ────────────────────────────────────────────────────────
def build_index(embedder, client):
    dim = embedder.get_embedding_dimension()

    # Check if already indexed
    try:
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME in collections:
            count = client.count(COLLECTION_NAME).count
            if count > 0:
                print(f"   Already indexed: {count} vectors. Skipping.")
                return
    except:
        pass

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )

    txt_files  = sorted(Path(ALERTS_DIR).glob("*.txt"))
    all_points = []

    print(f"   Indexing {len(txt_files)} alert files...")

    for i, fp in enumerate(txt_files):
        try:
            data   = parse_alert_file(str(fp))
            chunks = chunk_text(data["body"])
            embeds = embedder.encode(chunks, show_progress_bar=False)

            for j, (chunk, emb) in enumerate(zip(chunks, embeds)):
                all_points.append(PointStruct(
                    id      = str(uuid.uuid4()),
                    vector  = emb.tolist(),
                    payload = {
                        "alert_number": data["alert_number"],
                        "drug_name":    data["drug_name"],
                        "alert_type":   data["alert_type"],
                        "date":         data["date"],
                        "countries":    data["countries"],
                        "who_regions":  data["who_regions"],
                        "source_url":   data["source_url"],
                        "title":        data["title"],
                        "chunk_index":  j,
                        "text":         chunk,
                    }
                ))
            print(f"   [{i+1}/{len(txt_files)}] {fp.name} — {len(chunks)} chunks")
        except Exception as e:
            print(f"   Error {fp.name}: {e}")

    # Upload
    BATCH = 100
    for s in range(0, len(all_points), BATCH):
        client.upsert(COLLECTION_NAME, all_points[s:s+BATCH])

    print(f"\n   Total vectors indexed: {len(all_points)}")

# ── Retrieve ──────────────────────────────────────────────────────────────────
def retrieve(query, embedder, client, top_k=TOP_K):
    qvec = embedder.encode(query).tolist()
    return client.query_points(
        collection_name = COLLECTION_NAME,
        query           = qvec,
        limit           = top_k,
        with_payload    = True,
    ).points

# ── Build context string ──────────────────────────────────────────────────────
def build_context(results):
    parts = []
    for i, r in enumerate(results):
        p = r.payload
        parts.append(f"""--- SOURCE {i+1} (relevance: {r.score:.3f}) ---
Alert Number : {p.get('alert_number', 'N/A')}
Drug         : {p.get('drug_name', 'N/A')}
Alert Type   : {p.get('alert_type', 'N/A')}
Date         : {p.get('date', 'N/A')}
Countries    : {', '.join(p.get('countries', [])) or 'N/A'}
WHO Regions  : {', '.join(p.get('who_regions', [])) or 'N/A'}
URL          : {p.get('source_url', '')}

Content:
{p.get('text', '')}""")
    return "\n\n".join(parts)

# ── Ollama LLM call ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a pharmaceutical intelligence analyst specializing in counterfeit and substandard drug detection.
Answer using ONLY the WHO alert sources provided. 
Always cite the specific alert number for every claim (e.g. Alert 2/2024).
Never make up facts not present in the sources.
If sources are insufficient, say so clearly."""

def ask_ollama(query, context):
    full_prompt = f"""{SYSTEM_PROMPT}

SOURCES:
{context}

QUESTION: {query}

ANSWER (cite alert numbers):"""

    print("   Thinking...", end="", flush=True)
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model":  OLLAMA_MODEL,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 600,
                }
            },
            timeout=120
        )
        print(" done.")
        data = response.json()
        return data.get("response", "No response received.")
    except requests.exceptions.ConnectionError:
        print()
        return (
            "❌ Cannot connect to Ollama.\n"
            "Make sure Ollama is installed and running.\n"
            "Steps:\n"
            "  1. Install from https://ollama.com\n"
            "  2. Run: ollama pull mistral\n"
            "  3. Ollama runs automatically in background after install\n"
            "  4. Try running this script again"
        )
    except Exception as e:
        print()
        return f"❌ Ollama error: {e}"

# ── Display ───────────────────────────────────────────────────────────────────
def display(query, results, answer):
    print("\n" + "="*60)
    print(f"QUERY: {query}")
    print("="*60)
    print(f"\n📚 Top {len(results)} retrieved alerts:")
    for r in results:
        p = r.payload
        countries = ", ".join(p.get("countries", []))[:45]
        print(f"   [{r.score:.3f}] Alert {p.get('alert_number','?'):8} | {p.get('drug_name','?')[:30]:30} | {p.get('date','?'):15} | {countries}")
    print(f"\n🤖 Answer:")
    print("-"*60)
    print(answer)
    print("-"*60)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("Counterfeit Drug Intelligence Network")
    print("Powered by WHO Alerts + Ollama + Qdrant")
    print("="*60)

    # Load embedder
    print(f"\n[1/3] Loading multilingual embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)
    print(f"      Embedding dim: {embedder.get_embedding_dimension()}")

    # Qdrant (persistent — saves to disk)
    print(f"\n[2/3] Connecting to Qdrant (persistent storage)...")
    client = QdrantClient(path="qdrant_storage")
    print(f"      Storage path: qdrant_storage/")

    # Build index
    print(f"\n[3/3] Building index from {ALERTS_DIR}/...")
    build_index(embedder, client)

    # Demo queries
    demo_queries = [
        "Which falsified cancer medicines were found in Africa?",
        "What contaminated children medicines caused deaths?",
        "Tell me about fake COVID vaccines and where they were found",
        "What happened with falsified Ozempic semaglutide?",
        "Which substandard medicines were found in Pakistan or India?",
    ]

    print("\n" + "="*60)
    print("DEMO QUERIES")
    print("="*60)

    for query in demo_queries:
        results = retrieve(query, embedder, client)
        context = build_context(results)
        answer  = ask_ollama(query, context)
        display(query, results, answer)

    # Interactive
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("Ask anything about counterfeit/substandard medicines")
    print("Type 'exit' to quit")
    print("="*60)

    while True:
        try:
            query = input("\n🔍 Your question: ").strip()
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        if query.lower() in ["exit", "quit", "q", ""]:
            print("Goodbye.")
            break

        results = retrieve(query, embedder, client)
        context = build_context(results)
        answer  = ask_ollama(query, context)
        display(query, results, answer)


if __name__ == "__main__":
    main()
