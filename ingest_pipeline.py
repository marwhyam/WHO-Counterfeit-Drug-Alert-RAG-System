"""
WHO Alert Ingestion Pipeline
==============================
Reads all .txt files from who_alerts/, extracts structured entities,
chunks text, embeds using sentence-transformers, stores in Qdrant.

Install dependencies first:
    pip install qdrant-client sentence-transformers spacy
    python -m spacy download en_core_web_sm

Run:
    python ingest_pipeline.py
"""

import os
import json
import re
import uuid
from pathlib import Path
from datetime import datetime

# ── Dependencies ──────────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct, Filter,
        FieldCondition, MatchValue
    )
except ImportError:
    print("Run: pip install qdrant-client sentence-transformers")
    exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
ALERTS_DIR      = "who_alerts"
METADATA_FILE   = "who_alerts/metadata.json"
COLLECTION_NAME = "counterfeit_drug_alerts"
CHUNK_SIZE      = 400       # words per chunk
CHUNK_OVERLAP   = 80        # word overlap between chunks
EMBED_MODEL     = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# ↑ Multilingual model — handles English, French, Arabic, Urdu, Spanish
# This is critical for your cross-lingual RAG (XRAG paper requirement)

# ── Entity Extraction ─────────────────────────────────────────────────────────
# Regex patterns to extract structured fields from WHO alert text

COUNTRY_PATTERNS = [
    r"identified in ([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*)",
    r"detected in ([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*)",
    r"circulating in ([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*)",
    r"reported in ([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*)",
    r"found in ([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*)",
]

BATCH_PATTERNS = [
    r"[Bb]atch\s*(?:[Nn]o\.?|[Nn]umber|#)?\s*:?\s*([A-Z0-9\-\/]+)",
    r"[Ll]ot\s*(?:[Nn]o\.?|[Nn]umber|#)?\s*:?\s*([A-Z0-9\-\/]+)",
]

DRUG_PATTERNS = [
    r"[Ff]alsified\s+([A-Z][A-Za-z\s\(\)]+?)(?:\s+injection|\s+tablet|\s+capsule|\s+syrup|\s+for)",
    r"[Ss]ubstandard\s+(?:\(contaminated\)\s+)?([A-Za-z\s\(\)]+?)(?:\s+injection|\s+tablet|\s+capsule|\s+syrup|\s+identified)",
]

ALERT_TYPE_KEYWORDS = {
    "falsified":    ["falsified", "counterfeit", "fake"],
    "substandard":  ["substandard", "contaminated", "adulterated"],
    "both":         ["falsified", "substandard"],
}


def extract_entities(text: str, title: str) -> dict:
    """Extract structured metadata from alert text."""
    entities = {
        "drug_name":    "",
        "alert_type":   "",
        "countries":    [],
        "batch_numbers":[],
        "who_regions":  [],
    }

    # Drug name from title
    for pat in DRUG_PATTERNS:
        m = re.search(pat, title + " " + text[:500])
        if m:
            entities["drug_name"] = m.group(1).strip()
            break

    # If still empty, try extracting from title directly
    if not entities["drug_name"]:
        # e.g. "Medical Product Alert N°6/2025: Falsified SIMULECT..."
        m = re.search(r"(?:Falsified|Substandard)[^:]*:\s*(?:Falsified|Substandard)?\s*([A-Z][A-Za-z\s]+?)(?:\s+for|\s+injection|\s+tablet|$)", title)
        if m:
            entities["drug_name"] = m.group(1).strip()

    # Alert type
    text_lower = text.lower()
    if "falsified" in text_lower and "substandard" in text_lower:
        entities["alert_type"] = "both"
    elif "falsified" in text_lower or "counterfeit" in text_lower:
        entities["alert_type"] = "falsified"
    elif "substandard" in text_lower or "contaminated" in text_lower:
        entities["alert_type"] = "substandard"

    # Countries
    countries = set()
    for pat in COUNTRY_PATTERNS:
        for m in re.finditer(pat, text):
            for country in m.group(1).split(","):
                countries.add(country.strip())
    entities["countries"] = list(countries)

    # Batch numbers
    batches = set()
    for pat in BATCH_PATTERNS:
        for m in re.finditer(pat, text):
            batches.add(m.group(1).strip())
    entities["batch_numbers"] = list(batches)

    # WHO Regions
    regions = []
    region_map = {
        "African Region":           "AFRO",
        "European Region":          "EURO",
        "Eastern Mediterranean":    "EMRO",
        "South-East Asia":          "SEARO",
        "Western Pacific":          "WPRO",
        "Americas":                 "AMRO",
    }
    for region_name, code in region_map.items():
        if region_name.lower() in text_lower:
            regions.append(code)
    entities["who_regions"] = regions

    return entities


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping word-level chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


# ── Parse alert .txt file ─────────────────────────────────────────────────────
def parse_txt_file(filepath: str) -> dict:
    """Read a WHO alert .txt file and return structured data."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract header fields
    def extract_field(field_name):
        m = re.search(rf"^{field_name}:\s*(.+)$", content, re.MULTILINE)
        return m.group(1).strip() if m else ""

    alert_number = extract_field("ALERT NUMBER")
    title        = extract_field("TITLE")
    date_str     = extract_field("DATE")
    source_url   = extract_field("SOURCE URL")

    # Body text is everything after the === separator
    parts = content.split("=" * 60)
    body  = parts[1].strip() if len(parts) > 1 else content

    # Parse date
    parsed_date = ""
    for fmt in ["%Y-%m-%d", "%d %B %Y", "%B %d, %Y"]:
        try:
            parsed_date = datetime.strptime(date_str[:10], fmt[:len(date_str[:10])]).isoformat()
            break
        except:
            continue

    return {
        "alert_number": alert_number,
        "title":        title,
        "date":         parsed_date or date_str,
        "source_url":   source_url,
        "body":         body,
        "filename":     os.path.basename(filepath),
    }


# ── Main Ingestion ────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("WHO Alert Ingestion Pipeline")
    print("=" * 60)

    # 1. Load embedding model (multilingual — XRAG requirement)
    print(f"\n[1/5] Loading multilingual embedding model...")
    print(f"      Model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)
    embedding_dim = embedder.get_sentence_embedding_dimension()
    print(f"      Embedding dimension: {embedding_dim}")

    # 2. Connect to Qdrant (local in-memory for now)
    print(f"\n[2/5] Connecting to Qdrant...")
    client = QdrantClient(":memory:")   # Change to QdrantClient("localhost", port=6333) for persistent
    print(f"      Using in-memory Qdrant (change to localhost for persistence)")

    # 3. Create collection
    print(f"\n[3/5] Creating collection: {COLLECTION_NAME}")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=embedding_dim,
            distance=Distance.COSINE
        )
    )

    # 4. Process each alert file
    print(f"\n[4/5] Processing alerts from: {ALERTS_DIR}/")
    txt_files = sorted(Path(ALERTS_DIR).glob("*.txt"))
    print(f"      Found {len(txt_files)} alert files")

    all_points   = []
    all_metadata = []
    total_chunks = 0

    for i, filepath in enumerate(txt_files):
        print(f"\n  [{i+1}/{len(txt_files)}] {filepath.name}")

        try:
            # Parse file
            alert_data = parse_txt_file(str(filepath))

            # Extract entities
            entities = extract_entities(alert_data["body"], alert_data["title"])
            print(f"      Drug: {entities['drug_name'] or 'unknown'}")
            print(f"      Type: {entities['alert_type']}")
            print(f"      Countries: {entities['countries'][:3]}")
            print(f"      Batches: {entities['batch_numbers'][:3]}")

            # Chunk text
            chunks = chunk_text(alert_data["body"])
            print(f"      Chunks: {len(chunks)}")
            total_chunks += len(chunks)

            # Embed each chunk
            embeddings = embedder.encode(chunks, show_progress_bar=False)

            # Build Qdrant points
            for j, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = str(uuid.uuid4())
                payload  = {
                    # Core metadata
                    "alert_number":  alert_data["alert_number"],
                    "title":         alert_data["title"],
                    "date":          alert_data["date"],
                    "source_url":    alert_data["source_url"],
                    "filename":      alert_data["filename"],
                    "chunk_index":   j,
                    "chunk_total":   len(chunks),
                    # Extracted entities
                    "drug_name":     entities["drug_name"],
                    "alert_type":    entities["alert_type"],
                    "countries":     entities["countries"],
                    "batch_numbers": entities["batch_numbers"],
                    "who_regions":   entities["who_regions"],
                    # The actual text
                    "text":          chunk,
                }
                all_points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding.tolist(),
                        payload=payload
                    )
                )

            # Store metadata summary
            all_metadata.append({
                "alert_number":  alert_data["alert_number"],
                "title":         alert_data["title"],
                "date":          alert_data["date"],
                "drug_name":     entities["drug_name"],
                "alert_type":    entities["alert_type"],
                "countries":     entities["countries"],
                "batch_numbers": entities["batch_numbers"],
                "who_regions":   entities["who_regions"],
                "chunks":        len(chunks),
            })

        except Exception as e:
            print(f"      ERROR: {e}")
            continue

    # 5. Upload to Qdrant in batches
    print(f"\n[5/5] Uploading {len(all_points)} vectors to Qdrant...")
    BATCH = 100
    for start in range(0, len(all_points), BATCH):
        batch = all_points[start:start+BATCH]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        print(f"      Uploaded {min(start+BATCH, len(all_points))}/{len(all_points)}")

    # Save processed metadata
    with open("processed_metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"✅ Ingestion complete!")
    print(f"   Alerts processed : {len(all_metadata)}")
    print(f"   Total chunks     : {total_chunks}")
    print(f"   Vectors in Qdrant: {len(all_points)}")
    print(f"   Metadata saved   : processed_metadata.json")
    print(f"\nNext step: run query_pipeline.py to test retrieval")
    print(f"{'='*60}")

    # ── Quick test query ──────────────────────────────────────────────────────
    print(f"\n🔍 Running test query: 'falsified cancer medicine Africa'")
    query_vec = embedder.encode("falsified cancer medicine Africa").tolist()
    results   = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=3,
        with_payload=True
    )
    print(f"\nTop 3 results:")
    for r in results:
        print(f"\n  Score : {r.score:.4f}")
        print(f"  Alert : {r.payload.get('alert_number')}")
        print(f"  Drug  : {r.payload.get('drug_name')}")
        print(f"  Date  : {r.payload.get('date')}")
        print(f"  Text  : {r.payload.get('text', '')[:150]}...")


if __name__ == "__main__":
    main()
