"""
WHO Alert Downloader v2 — Fixed
================================
The first scraper got empty bodies because WHO pages load content via JS.
This version:
1. Extracts all useful info from the title + URL (drug, type, date)
2. Fetches the actual page with a better parser targeting WHO's content divs
3. Falls back gracefully if content still empty

Run:
    pip install requests beautifulsoup4
    python who_downloader_v2.py

Output: who_alerts_v2/ folder with enriched .txt files
"""

import requests
import json
import time
import os
import re
from bs4 import BeautifulSoup
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
EXISTING_META = "who_alerts/metadata.json"   # from your first download
OUT_DIR       = "who_alerts_v2"
HEADERS       = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
DELAY = 2.0

os.makedirs(OUT_DIR, exist_ok=True)

# ── Extract info from title ───────────────────────────────────────────────────
def parse_title(title: str) -> dict:
    """Extract date, alert number, drug name, alert type from title string."""
    
    # Clean the messy title format
    # e.g. "24 January 2013Medical product alertMedical Product Alert N°1/2013: Contaminated..."
    
    # Extract alert number
    alert_num = ""
    m = re.search(r"N[°o]?\s*(\d+/\d+)", title)
    if m:
        alert_num = m.group(1)

    # Extract date
    date = ""
    m = re.search(r"(\d{1,2}\s+\w+\s+\d{4})", title)
    if m:
        date = m.group(1)

    # Extract drug/product name — everything after the colon
    drug = ""
    m = re.search(r"N[°o]?\s*\d+/\d+[:\-]\s*(.+)$", title)
    if m:
        drug = m.group(1).strip()
        # Remove common prefixes
        drug = re.sub(r"^(Falsified|Substandard|Contaminated|Adverse reactions caused by)\s+", "", drug, flags=re.IGNORECASE).strip()
        drug = re.sub(r"\s+(UPDATE|update)$", "", drug).strip()

    # Alert type from title
    alert_type = ""
    title_lower = title.lower()
    if "falsified" in title_lower or "counterfeit" in title_lower:
        alert_type = "falsified"
    if "substandard" in title_lower or "contaminated" in title_lower:
        alert_type = "substandard" if not alert_type else "both"

    return {
        "alert_number": alert_num,
        "date":         date,
        "drug_name":    drug,
        "alert_type":   alert_type,
    }


def extract_url_info(url: str) -> dict:
    """Extract date and keywords from the URL slug."""
    slug = url.split("/news/item/")[-1] if "/news/item/" in url else ""
    
    # Date from URL: e.g. 24-01-2013
    date = ""
    m = re.search(r"(\d{2}-\d{2}-\d{4})", slug)
    if m:
        parts = m.group(1).split("-")
        months = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
        try:
            d, mo, y = parts
            month_name = months[int(mo)-1].capitalize()
            date = f"{int(d)} {month_name} {y}"
        except:
            date = m.group(1)
    
    return {"url_date": date, "slug": slug}


def fetch_who_page(url: str) -> str:
    """Try to fetch real content from WHO alert page."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # WHO content is usually in these containers
        content = ""
        for selector in [
            "div.sf-detail-body-wrapper",
            "div.content-wrapper",
            "div.sf-content",
            "article",
            "div.item-page",
            "div#content",
            "main",
        ]:
            el = soup.select_one(selector)
            if el:
                # Remove navigation, headers, footers
                for tag in el.select("nav, header, footer, .sf-nav, script, style"):
                    tag.decompose()
                content = el.get_text(separator="\n", strip=True)
                if len(content) > 200:
                    break

        # If still nothing useful, get all paragraphs
        if len(content) < 200:
            paras = soup.find_all("p")
            content = "\n".join(p.get_text(strip=True) for p in paras if len(p.get_text(strip=True)) > 30)

        return content[:10000] if content else ""

    except Exception as e:
        return f"[Fetch error: {e}]"


def save_alert(data: dict, filename: str):
    """Save enriched alert as structured text."""
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"ALERT NUMBER: {data.get('alert_number','')}\n")
        f.write(f"DRUG NAME: {data.get('drug_name','')}\n")
        f.write(f"ALERT TYPE: {data.get('alert_type','')}\n")
        f.write(f"DATE: {data.get('date','')}\n")
        f.write(f"COUNTRIES: {', '.join(data.get('countries', []))}\n")
        f.write(f"WHO REGIONS: {', '.join(data.get('who_regions', []))}\n")
        f.write(f"SOURCE URL: {data.get('source_url','')}\n")
        f.write(f"TITLE: {data.get('title','')}\n")
        f.write("\n" + "="*60 + "\n\n")
        
        body = data.get("body", "")
        if body and len(body) > 100:
            f.write(body)
        else:
            # Construct synthetic body from title metadata — still useful for RAG
            f.write(f"WHO Medical Product Alert {data.get('alert_number','')}.\n\n")
            f.write(f"Date: {data.get('date','')}\n")
            f.write(f"Product: {data.get('drug_name','')}\n")
            f.write(f"Alert Type: {data.get('alert_type','')}\n")
            f.write(f"This alert concerns {data.get('alert_type','')} {data.get('drug_name','')} ")
            f.write(f"reported to the World Health Organization.\n")
            f.write(f"Source: {data.get('source_url','')}\n")
    return path


def detect_countries_regions(text: str) -> dict:
    """Detect country and region mentions in text."""
    text_lower = text.lower()
    
    country_list = [
        "Nigeria","Ghana","Kenya","Uganda","Tanzania","Ethiopia","Cameroon",
        "Senegal","Mali","Niger","Chad","Sudan","Somalia","Rwanda","Zambia",
        "Zimbabwe","Mozambique","Angola","Madagascar","Malawi","Gambia",
        "Guinea","Liberia","Sierra Leone","Ivory Coast","Burkina Faso","Togo","Benin",
        "India","Pakistan","Bangladesh","Indonesia","Philippines","Vietnam",
        "Thailand","Myanmar","Cambodia","Laos","Nepal","Sri Lanka","Malaysia",
        "France","Germany","Italy","Spain","Turkey","Ukraine","Poland",
        "Romania","Bulgaria","Greece","Serbia","Croatia","Hungary",
        "United States","United Kingdom","Canada","Australia","New Zealand",
        "Brazil","Mexico","Argentina","Colombia","Peru","Chile","Ecuador",
        "China","Japan","South Korea","Singapore","Taiwan",
        "Iraq","Iran","Syria","Lebanon","Jordan","Egypt","Libya",
        "Morocco","Algeria","Tunisia","Saudi Arabia","UAE","Qatar","Kuwait",
        "Congo","Burundi","Eritrea","Djibouti","Comoros",
    ]
    
    countries = [c for c in country_list if c.lower() in text_lower]
    
    region_map = {
        "african region": "AFRO", "africa": "AFRO", "sub-saharan": "AFRO",
        "european region": "EURO", "europe": "EURO",
        "eastern mediterranean": "EMRO", "middle east": "EMRO",
        "south-east asia": "SEARO", "southeast asia": "SEARO", "south east asia": "SEARO",
        "western pacific": "WPRO", "pacific": "WPRO",
        "americas": "AMRO", "latin america": "AMRO",
    }
    regions = list({code for key, code in region_map.items() if key in text_lower})
    
    return {"countries": list(set(countries)), "who_regions": regions}


def main():
    # Load existing metadata from first download
    with open(EXISTING_META, "r", encoding="utf-8") as f:
        existing = json.load(f)
    
    print(f"Loaded {len(existing)} alerts from existing metadata")
    print(f"Output directory: {OUT_DIR}/\n")
    
    all_enriched = []

    for i, alert in enumerate(existing):
        title      = alert.get("title", "")
        source_url = alert.get("url", "")
        filename_base = f"{alert.get('alert_number','').replace('/','-') or str(i)}"
        filename   = f"{filename_base}.txt"

        print(f"[{i+1}/{len(existing)}] Alert {alert.get('alert_number','')} — {title[:50]}...")

        # Extract from title
        parsed = parse_title(title)
        url_info = extract_url_info(source_url)
        
        # Use URL date if title date missing
        if not parsed["date"] and url_info["url_date"]:
            parsed["date"] = url_info["url_date"]

        # Try to fetch real page content
        print(f"    Fetching: {source_url[:60]}...")
        body = fetch_who_page(source_url)
        
        if len(body) > 200:
            print(f"    Got {len(body)} chars of content ✓")
        else:
            print(f"    Content empty — using title metadata only")

        # Detect countries/regions from body + title
        geo = detect_countries_regions(body + " " + title)

        enriched = {
            "alert_number": parsed["alert_number"],
            "drug_name":    parsed["drug_name"],
            "alert_type":   parsed["alert_type"],
            "date":         parsed["date"],
            "countries":    geo["countries"],
            "who_regions":  geo["who_regions"],
            "source_url":   source_url,
            "title":        title,
            "body":         body,
        }

        saved = save_alert(enriched, filename)
        enriched["filename"] = filename
        all_enriched.append(enriched)

        print(f"    Drug: {parsed['drug_name'] or 'unknown'} | Type: {parsed['alert_type']} | Countries: {geo['countries'][:3]}")
        print(f"    Saved → {saved}")

        # Save progress
        with open(f"{OUT_DIR}/enriched_metadata.json", "w", encoding="utf-8") as f:
            json.dump(all_enriched, f, indent=2, ensure_ascii=False)

        time.sleep(DELAY)

    print(f"\n✅ Done! {len(all_enriched)} enriched alerts saved to {OUT_DIR}/")
    print(f"   Metadata: {OUT_DIR}/enriched_metadata.json")
    print(f"\nNext: run ingest_pipeline_v3.py with the new who_alerts_v2/ folder")


if __name__ == "__main__":
    main()
