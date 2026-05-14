"""
Hyper-RAG: Hallucination Control Layer
========================================
Based on: Hyper-RAG arxiv 2504.08758
Published: Nature Communications 2026

Core idea from the paper:
- Standard RAG retrieves chunks independently (pairwise relationships)
- Hyper-RAG builds hyperedges connecting MULTIPLE chunks that share entities
- Before generating an answer, cross-verify claims across hyperedges
- If a fact appears in only 1 source = LOW confidence
- If a fact appears in 2+ sources = HIGH confidence
- Refuse to state facts with LOW confidence (hallucination prevention)

Results from paper: +35.5% over LightRAG, +12.3% over direct LLM use
Especially effective for high-stakes domains (medical, drug safety)

Install:
    pip install qdrant-client sentence-transformers requests networkx
    python hyper_rag.py
"""

import os, re, json, uuid
from pathlib import Path
from collections import defaultdict
import requests
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# ── Config ────────────────────────────────────────────────────────────────────
ALERTS_DIR      = "who_alerts_v2"
COLLECTION_NAME = "counterfeit_drug_alerts"
EMBED_MODEL     = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_MODEL    = "tinyllama"
TIMEOUT         = 300
MIN_CONFIDENCE  = 2   # minimum sources needed to state a fact confidently

# ── Parse alert ───────────────────────────────────────────────────────────────
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

    return {
        "alert_number": field("ALERT NUMBER"),
        "drug_name":    field("DRUG NAME"),
        "alert_type":   field("ALERT TYPE"),
        "date":         field("DATE"),
        "year":         year,
        "countries":    [c.strip() for c in field("COUNTRIES").split(",") if c.strip()],
        "who_regions":  [r.strip() for r in field("WHO REGIONS").split(",") if r.strip()],
        "source_url":   field("SOURCE URL"),
        "body":         body,
        "filename":     os.path.basename(filepath),
    }

# ────────────────────────────────────────────────────────────────────────────
# HYPER-RAG CORE: Build Hyperedges
#
# Paper: "Hyper-RAG captures both pairwise AND beyond-pairwise correlations
# in domain-specific knowledge"
#
# A hyperedge connects multiple chunks that share the same entity.
# Example hyperedge: {Alert 2/2013, Alert 5/2013} both mention "Coartem"
#                    {Alert 7/2022, Alert 6/2022} both mention "paediatric"
#
# This is the key difference from standard GraphRAG which only does
# pairwise edges (A→B), not hyperedges (A,B,C all share entity X)
# ────────────────────────────────────────────────────────────────────────────

class HyperedgeIndex:
    def __init__(self):
        # entity -> set of alert_numbers that mention it
        self.entity_to_alerts  = defaultdict(set)
        # alert -> set of entities it contains
        self.alert_to_entities = defaultdict(set)
        # hyperedge_id -> frozenset of alert_numbers
        self.hyperedges        = {}
        # entity -> hyperedge_ids
        self.entity_to_hyperedges = defaultdict(set)

    def add_alert(self, alert):
        alert_num = alert["alert_number"]
        if not alert_num:
            return

        # Extract entities from this alert
        entities = self._extract_entities(alert)

        for entity in entities:
            self.entity_to_alerts[entity].add(alert_num)
            self.alert_to_entities[alert_num].add(entity)

    def _extract_entities(self, alert):
        """Extract normalized entities from an alert."""
        entities = set()

        # Drug name — normalize
        drug = alert.get("drug_name", "").lower().strip()
        if drug and drug != "unknown":
            # Extract core drug name (remove dosage info)
            core = re.sub(r"\s*\(.*?\)", "", drug).strip()
            core = re.sub(r"\s+\d+mg.*", "", core).strip()
            if len(core) > 2:
                entities.add(f"DRUG:{core}")

        # Countries
        for country in alert.get("countries", []):
            entities.add(f"COUNTRY:{country.lower()}")

        # Alert type
        atype = alert.get("alert_type", "")
        if atype:
            entities.add(f"TYPE:{atype}")

        # WHO regions
        for region in alert.get("who_regions", []):
            entities.add(f"REGION:{region}")

        # Year
        year = alert.get("year")
        if year:
            entities.add(f"YEAR:{year}")

        # Keywords from body
        body = alert.get("body", "").lower()
        medical_keywords = [
            "malaria", "antimalarial", "vaccine", "paediatric", "children",
            "cancer", "insulin", "antibiotic", "contaminated", "falsified",
            "covid", "hepatitis", "tuberculosis", "hiv", "syrup",
            "injection", "tablet", "capsule", "opioid", "fentanyl",
        ]
        for kw in medical_keywords:
            if kw in body:
                entities.add(f"KEYWORD:{kw}")

        return entities

    def build_hyperedges(self):
        """
        Build hyperedges: for each entity shared by 2+ alerts,
        create a hyperedge connecting all those alerts.
        """
        he_id = 0
        for entity, alert_set in self.entity_to_alerts.items():
            if len(alert_set) >= 2:  # hyperedge needs 2+ members
                hid = f"HE_{he_id}"
                self.hyperedges[hid] = {
                    "id":        hid,
                    "entity":    entity,
                    "alerts":    frozenset(alert_set),
                    "strength":  len(alert_set),  # more alerts = stronger hyperedge
                }
                self.entity_to_hyperedges[entity].add(hid)
                he_id += 1

        return len(self.hyperedges)

    def get_hyperedge_context(self, alert_numbers):
        """
        Given a set of retrieved alert numbers, find all hyperedges
        connecting them and return the shared entities (cross-verified facts).
        """
        alert_set    = set(alert_numbers)
        shared       = defaultdict(list)  # entity -> list of alert_nums that confirm it
        hyperedge_hits = []

        for hid, he in self.hyperedges.items():
            # Check how many of our retrieved alerts are in this hyperedge
            overlap = alert_set & he["alerts"]
            if len(overlap) >= 2:
                hyperedge_hits.append({
                    "hyperedge_id": hid,
                    "entity":       he["entity"],
                    "confirmed_by": list(overlap),
                    "strength":     len(overlap),
                })
                shared[he["entity"]].extend(list(overlap))

        # Sort by strength
        hyperedge_hits.sort(key=lambda x: x["strength"], reverse=True)

        return hyperedge_hits, shared

    def get_confidence_score(self, claim_entity, retrieved_alerts):
        """
        How many independent sources confirm this entity/claim?
        This is the core hallucination prevention mechanism.
        """
        alert_set = set(retrieved_alerts)
        confirming = self.entity_to_alerts.get(claim_entity, set()) & alert_set
        return len(confirming)


# ────────────────────────────────────────────────────────────────────────────
# CROSS-SOURCE VERIFICATION
# Paper: "cross-references multiple sources before providing an answer"
# ────────────────────────────────────────────────────────────────────────────

def verify_and_grade(retrieved_results, hyperedge_index):
    """
    Grade retrieved results by cross-source verification.
    Returns verified facts and confidence scores.
    """
    alert_numbers = [r.payload.get("alert_number", "") for r in retrieved_results]
    alert_numbers = [a for a in alert_numbers if a]

    # Get hyperedge context
    hyperedge_hits, shared_entities = hyperedge_index.get_hyperedge_context(alert_numbers)

    # Build verification report
    verified_facts   = []  # facts confirmed by 2+ sources
    unverified_facts = []  # facts from only 1 source

    for entity, confirming_alerts in shared_entities.items():
        count = len(set(confirming_alerts))
        fact  = {
            "entity":           entity,
            "confirmed_by":     list(set(confirming_alerts)),
            "confidence":       "HIGH" if count >= MIN_CONFIDENCE else "LOW",
            "source_count":     count,
        }
        if count >= MIN_CONFIDENCE:
            verified_facts.append(fact)
        else:
            unverified_facts.append(fact)

    return {
        "verified_facts":   sorted(verified_facts,   key=lambda x: x["source_count"], reverse=True),
        "unverified_facts": sorted(unverified_facts, key=lambda x: x["source_count"], reverse=True),
        "hyperedge_hits":   hyperedge_hits,
        "alert_count":      len(alert_numbers),
    }


# ────────────────────────────────────────────────────────────────────────────
# GROUNDED ANSWER GENERATION
# Only state facts that are cross-verified by 2+ sources
# ────────────────────────────────────────────────────────────────────────────

def generate_grounded_answer(query, retrieved_results, verification, alerts_by_num):
    """Generate answer with hallucination grounding."""

    # Build verified context string
    verified_entities = [f["entity"] for f in verification["verified_facts"][:10]]

    context_parts = []
    for r in retrieved_results:
        p = r.payload
        context_parts.append(
            f"[Alert {p.get('alert_number','?')} | "
            f"{p.get('drug_name','?')} | "
            f"{p.get('date','?')} | "
            f"Countries: {', '.join(p.get('countries',[]))}]\n"
            f"{p.get('text','')[:300]}"
        )
    context = "\n\n".join(context_parts)

    # Build verification summary for LLM
    verified_summary = ""
    if verification["verified_facts"]:
        verified_summary = "CROSS-VERIFIED FACTS (confirmed by 2+ sources — state with HIGH confidence):\n"
        for fact in verification["verified_facts"][:8]:
            entity_clean = fact["entity"].replace("DRUG:", "Drug: ").replace(
                "COUNTRY:", "Country: ").replace("KEYWORD:", "Topic: ").replace(
                "YEAR:", "Year: ").replace("TYPE:", "Alert type: ")
            verified_summary += f"  ✓ {entity_clean} (confirmed by {fact['source_count']} alerts)\n"

    unverified_summary = ""
    if verification["unverified_facts"]:
        unverified_summary = "\nSINGLE-SOURCE FACTS (only 1 source — mention with caution or omit):\n"
        for fact in verification["unverified_facts"][:5]:
            entity_clean = fact["entity"].replace("DRUG:", "Drug: ").replace(
                "COUNTRY:", "Country: ")
            unverified_summary += f"  ⚠ {entity_clean}\n"

    prompt = f"""You are a WHO pharmaceutical intelligence analyst.
Answer the question using ONLY the provided sources.
IMPORTANT: Only state facts marked ✓ with HIGH confidence. 
Facts marked ⚠ should be omitted or mentioned with uncertainty.
Always cite alert numbers.

{verified_summary}
{unverified_summary}

SOURCES:
{context[:1800]}

QUESTION: {query}

GROUNDED ANSWER (cite alerts, only state verified facts):"""

    print("   Generating grounded answer", end="", flush=True)
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 400, "num_ctx": 2048}
            },
            timeout=TIMEOUT
        )
        print(" ✓")
        return r.json().get("response", "No response.").strip()
    except Exception as e:
        print()
        return f"❌ Ollama error: {e}"


# ── Display verification report ───────────────────────────────────────────────
def display_verification(verification):
    print(f"\n🔬 Hyper-RAG Verification Report:")
    print(f"   Hyperedges activated : {len(verification['hyperedge_hits'])}")
    print(f"   HIGH confidence facts: {len(verification['verified_facts'])}")
    print(f"   LOW confidence facts : {len(verification['unverified_facts'])}")

    if verification["verified_facts"]:
        print(f"\n   ✅ Cross-verified (safe to state):")
        for fact in verification["verified_facts"][:6]:
            entity_clean = fact["entity"].replace("DRUG:", "Drug: ").replace(
                "COUNTRY:", "Country: ").replace("KEYWORD:", "Topic: ").replace(
                "YEAR:", "Year: ").replace("TYPE:", "Alert type: ")
            alerts_str = ", ".join(fact["confirmed_by"][:3])
            print(f"      [{fact['source_count']} sources] {entity_clean} — confirmed by Alert {alerts_str}")

    if verification["unverified_facts"][:3]:
        print(f"\n   ⚠️  Single-source (hallucination risk):")
        for fact in verification["unverified_facts"][:3]:
            entity_clean = fact["entity"].replace("DRUG:", "Drug: ").replace(
                "COUNTRY:", "Country: ")
            print(f"      [1 source] {entity_clean}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("Hyper-RAG: Hallucination Control Layer")
    print("Based on arxiv 2504.08758 (Nature Communications 2026)")
    print("="*60)

    # Load alerts
    print("\n[1/5] Loading alerts...")
    txt_files = sorted(Path(ALERTS_DIR).glob("*.txt"))
    alerts    = [parse_alert_file(str(fp)) for fp in txt_files]
    alerts_by_num = {a["alert_number"]: a for a in alerts if a["alert_number"]}
    print(f"   Loaded {len(alerts)} alerts")

    # Build hyperedge index
    print("\n[2/5] Building Hyperedge Index...")
    heidx = HyperedgeIndex()
    for alert in alerts:
        heidx.add_alert(alert)
    n_hyperedges = heidx.build_hyperedges()
    print(f"   Entities indexed  : {len(heidx.entity_to_alerts)}")
    print(f"   Hyperedges built  : {n_hyperedges}")

    # Show top hyperedges
    top_he = sorted(heidx.hyperedges.values(), key=lambda x: x["strength"], reverse=True)[:5]
    print(f"\n   Strongest hyperedges (most cross-referenced entities):")
    for he in top_he:
        entity_clean = he["entity"].replace("DRUG:", "Drug: ").replace(
            "COUNTRY:", "Country: ").replace("KEYWORD:", "Topic: ").replace(
            "YEAR:", "Year: ").replace("TYPE:", "Alert type: ")
        print(f"      [{he['strength']} alerts] {entity_clean}")

    # Load embedder
    print("\n[3/5] Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)
    print(f"   Dim: {embedder.get_embedding_dimension()}")

    # Load Qdrant
    print("\n[4/5] Connecting to Qdrant...")
    client = QdrantClient(path="qdrant_storage")
    print(f"   Vectors: {client.count(COLLECTION_NAME).count}")

    # Warm up Ollama
    print("\n[5/5] Warming up TinyLlama...")
    try:
        requests.post(OLLAMA_URL,
                     json={"model": OLLAMA_MODEL, "prompt": "Hi",
                           "stream": False, "options": {"num_predict": 3}},
                     timeout=60)
        print("   Warmed up ✓")
    except:
        print("   Ollama not responding")

    # ── Demo queries ──────────────────────────────────────────────────────────
    demo_queries = [
        "Which falsified cancer medicines were found in Africa?",
        "What contaminated paediatric medicines caused child deaths?",
        "Tell me about fake COVID vaccines found in 2021",
        "Which drugs were falsified in Pakistan?",
    ]

    print("\n" + "="*60)
    print("HYPER-RAG GROUNDED QUERIES")
    print("="*60)

    for query in demo_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")

        # Retrieve
        qvec    = embedder.encode(query).tolist()
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=qvec, limit=6, with_payload=True
        ).points

        print(f"\n📚 Retrieved {len(results)} alerts:")
        for r in results:
            p = r.payload
            countries = ", ".join(p.get("countries", []))[:35]
            print(f"   [{r.score:.3f}] Alert {p.get('alert_number','?'):8} | "
                  f"{p.get('drug_name','?')[:25]:25} | {countries}")

        # Hyper-RAG verification
        verification = verify_and_grade(results, heidx)
        display_verification(verification)

        # Grounded answer
        answer = generate_grounded_answer(query, results, verification, alerts_by_num)
        print(f"\n🤖 Grounded Answer:")
        print("-"*60)
        print(answer)
        print("-"*60)

    # ── Interactive ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("Every answer is cross-verified before generation.")
    print("Type 'exit' to quit")
    print("="*60)

    while True:
        try:
            query = input("\n🔍 Question: ").strip()
        except KeyboardInterrupt:
            break
        if query.lower() in ["exit", "quit", "q", ""]:
            break

        qvec    = embedder.encode(query).tolist()
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=qvec, limit=6, with_payload=True
        ).points

        print(f"\n📚 Retrieved {len(results)} alerts:")
        for r in results:
            p = r.payload
            print(f"   [{r.score:.3f}] Alert {p.get('alert_number','?'):8} | "
                  f"{p.get('drug_name','?')[:30]:30} | "
                  f"{', '.join(p.get('countries',[]))[:30]}")

        verification = verify_and_grade(results, heidx)
        display_verification(verification)

        answer = generate_grounded_answer(query, results, verification, alerts_by_num)
        print(f"\n🤖 Grounded Answer:")
        print("-"*60)
        print(answer)
        print("-"*60)

    print("Done.")

if __name__ == "__main__":
    main()
