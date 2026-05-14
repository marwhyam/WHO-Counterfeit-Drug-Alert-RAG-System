"""
Counterfeit Drug Intelligence — RAG Query Pipeline (TinyLlama)
===============================================================
Uses TinyLlama — fast, small, works on any PC.

SETUP:
    ollama pull tinyllama
    pip install qdrant-client sentence-transformers requests
    python rag_query_pipeline_tiny.py
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
OLLAMA_MODEL    = "tinyllama"    # fast and light
TIMEOUT         = 300            # 5 minutes — enough for any PC

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

    BATCH = 100
    for s in range(0, len(all_points), BATCH):
        client.upsert(COLLECTION_NAME, all_points[s:s+BATCH])
    print(f"   Total vectors: {len(all_points)}")

# ── Retrieve ──────────────────────────────────────────────────────────────────
def retrieve(query, embedder, client, top_k=TOP_K):
    qvec = embedder.encode(query).tolist()
    return client.query_points(
        collection_name = COLLECTION_NAME,
        query           = qvec,
        limit           = top_k,
        with_payload    = True,
    ).points

# ── Build context ─────────────────────────────────────────────────────────────
def build_context(results):
    parts = []
    for i, r in enumerate(results):
        p = r.payload
        parts.append(
            f"[SOURCE {i+1} | Alert {p.get('alert_number','?')} | "
            f"{p.get('drug_name','?')} | {p.get('date','?')} | "
            f"Countries: {', '.join(p.get('countries',[])) or 'unknown'}]\n"
            f"{p.get('text','')}"
        )
    return "\n\n".join(parts)

# ── Ollama call ───────────────────────────────────────────────────────────────
def ask_ollama(query, context):
    # Keep prompt short for tinyllama
    prompt = f"""You are a drug safety analyst. Using ONLY these WHO alert sources, answer the question. Cite alert numbers.

SOURCES:
{context[:2000]}

QUESTION: {query}

ANSWER:"""

    print("   Generating answer", end="", flush=True)
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 300,
                    "num_ctx":     2048,
                }
            },
            timeout=TIMEOUT
        )
        print(" ✓")
        return response.json().get("response", "No response.").strip()
    except requests.exceptions.ConnectionError:
        print()
        return "❌ Ollama not running. Run: ollama serve"
    except requests.exceptions.Timeout:
        print()
        return "❌ Timeout — your PC may need more time. Try running: ollama run tinyllama (to warm it up first)"
    except Exception as e:
        print()
        return f"❌ Error: {e}"

# ── Display ───────────────────────────────────────────────────────────────────
def display(query, results, answer):
    print("\n" + "="*60)
    print(f"QUERY: {query}")
    print("="*60)
    print(f"\n📚 Retrieved alerts:")
    for r in results:
        p = r.payload
        countries = ", ".join(p.get("countries", []))[:40]
        print(f"   [{r.score:.3f}] Alert {p.get('alert_number','?'):10} | "
              f"{p.get('drug_name','?')[:28]:28} | {countries}")
    print(f"\n🤖 Answer:")
    print("-"*60)
    print(answer)
    print("-"*60)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("Counterfeit Drug Intelligence Network")
    print("WHO Alerts + TinyLlama + Qdrant")
    print("="*60)

    print(f"\n[1/3] Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)
    print(f"      Dim: {embedder.get_embedding_dimension()}")

    print(f"\n[2/3] Connecting to Qdrant...")
    client = QdrantClient(path="qdrant_storage")

    print(f"\n[3/3] Building index...")
    build_index(embedder, client)

    # Warm up ollama before queries
    print("\nWarming up TinyLlama (first call may take 30 sec)...")
    warmup = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": "Hello", "stream": False,
              "options": {"num_predict": 5}},
        timeout=TIMEOUT
    )
    print("Warmed up ✓")

    demo_queries = [
        "Which falsified cancer medicines were found in Africa?",
        "What contaminated children medicines caused deaths?",
        "Tell me about fake COVID vaccines",
        "What happened with falsified Ozempic semaglutide?",
        "Which substandard medicines were found in Pakistan?",
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
    print("INTERACTIVE MODE — type your questions (or 'exit')")
    print("="*60)

    while True:
        try:
            query = input("\n🔍 Question: ").strip()
        except KeyboardInterrupt:
            break
        if query.lower() in ["exit", "quit", "q", ""]:
            break
        results = retrieve(query, embedder, client)
        context = build_context(results)
        answer  = ask_ollama(query, context)
        display(query, results, answer)

    print("Done.")

if __name__ == "__main__":
    main()
