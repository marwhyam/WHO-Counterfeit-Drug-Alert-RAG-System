"""
WHO Counterfeit Drug Alert Downloader
======================================
Run this on YOUR machine:
    pip install requests beautifulsoup4
    python who_alert_downloader.py

Downloads all WHO Medical Product Alerts as PDFs + saves metadata as JSON.
This is your RAG corpus for the Counterfeit Drug Intelligence project.
"""

import requests
import json
import time
import os
import re
from bs4 import BeautifulSoup
from datetime import datetime

# ── Config ──────────────────────────────────────────────────────────────────
BASE_URL   = "https://www.who.int"
LIST_URL   = "https://www.who.int/teams/regulation-prequalification/incidents-and-SF/full-list-of-who-medical-product-alerts"
OUT_DIR    = "who_alerts"
META_FILE  = "who_alerts/metadata.json"
HEADERS    = {"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
DELAY      = 1.5   # seconds between requests — be polite to WHO servers

os.makedirs(OUT_DIR, exist_ok=True)

# ── Step 1: Scrape the full alert list page ──────────────────────────────────
def get_alert_links():
    print("Fetching WHO alert list...")
    r = requests.get(LIST_URL, headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    alerts = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True)
        # WHO alert links contain /news/item/ in their path
        if "/news/item/" in href and "medical-product-alert" in href.lower():
            full_url = href if href.startswith("http") else BASE_URL + href
            alerts.append({"title": text, "url": full_url})

    # Deduplicate
    seen = set()
    unique = []
    for a in alerts:
        if a["url"] not in seen:
            seen.add(a["url"])
            unique.append(a)

    print(f"Found {len(unique)} alerts.")
    return unique


# ── Step 2: For each alert page, extract metadata + PDF link ─────────────────
def parse_alert_page(alert):
    r = requests.get(alert["url"], headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Extract body text
    body = ""
    content_div = soup.find("div", class_=re.compile(r"sf-content|article|content"))
    if content_div:
        body = content_div.get_text(separator="\n", strip=True)
    else:
        body = soup.get_text(separator="\n", strip=True)[:5000]

    # Extract PDF link if exists
    pdf_url = None
    for a in soup.find_all("a", href=True):
        if a["href"].endswith(".pdf"):
            pdf_url = a["href"] if a["href"].startswith("http") else BASE_URL + a["href"]
            break

    # Extract date from meta or URL
    date_str = ""
    meta_date = soup.find("meta", {"name": "date"}) or soup.find("meta", {"property": "article:published_time"})
    if meta_date:
        date_str = meta_date.get("content", "")

    # Extract drug name, countries, alert number from title
    title = alert["title"]
    alert_num = re.search(r"N[°o]?\s*(\d+/\d+)", title, re.IGNORECASE)
    alert_number = alert_num.group(1) if alert_num else ""

    return {
        "alert_number": alert_number,
        "title": title,
        "url": alert["url"],
        "date": date_str,
        "pdf_url": pdf_url,
        "body_text": body[:8000],   # first 8000 chars for RAG chunking
    }


# ── Step 3: Download PDF ─────────────────────────────────────────────────────
def download_pdf(pdf_url, filename):
    try:
        r = requests.get(pdf_url, headers=HEADERS, timeout=30, stream=True)
        r.raise_for_status()
        filepath = os.path.join(OUT_DIR, filename)
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  Downloaded PDF → {filepath}")
        return filepath
    except Exception as e:
        print(f"  PDF download failed: {e}")
        return None


# ── Step 4: Save full text as .txt for RAG ingestion ────────────────────────
def save_text(alert_data):
    safe_name = re.sub(r"[^\w\-]", "_", alert_data["alert_number"] or alert_data["title"][:40])
    txt_path = os.path.join(OUT_DIR, f"{safe_name}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"ALERT NUMBER: {alert_data['alert_number']}\n")
        f.write(f"TITLE: {alert_data['title']}\n")
        f.write(f"DATE: {alert_data['date']}\n")
        f.write(f"SOURCE URL: {alert_data['url']}\n")
        f.write(f"PDF: {alert_data['pdf_url']}\n")
        f.write("\n" + "="*60 + "\n\n")
        f.write(alert_data["body_text"])
    return txt_path


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    all_metadata = []

    # Get list of all alerts
    alerts = get_alert_links()

    for i, alert in enumerate(alerts):
        print(f"\n[{i+1}/{len(alerts)}] Processing: {alert['title'][:70]}")
        try:
            # Parse the alert page
            data = parse_alert_page(alert)

            # Save text version
            txt_path = save_text(data)
            data["txt_path"] = txt_path
            print(f"  Saved text → {txt_path}")

            # Download PDF if available
            if data["pdf_url"]:
                safe_name = re.sub(r"[^\w\-]", "_", data["alert_number"] or str(i))
                pdf_path = download_pdf(data["pdf_url"], f"{safe_name}.pdf")
                data["pdf_path"] = pdf_path
            else:
                data["pdf_path"] = None
                print("  No PDF found — text only")

            all_metadata.append(data)

            # Save metadata progressively (so if it crashes, you keep progress)
            with open(META_FILE, "w", encoding="utf-8") as f:
                json.dump(all_metadata, f, indent=2, ensure_ascii=False)

            time.sleep(DELAY)

        except Exception as e:
            print(f"  ERROR on {alert['url']}: {e}")
            continue

    print(f"\n✅ Done! Downloaded {len(all_metadata)} alerts.")
    print(f"   Text files + PDFs saved to: {OUT_DIR}/")
    print(f"   Metadata saved to: {META_FILE}")
    print(f"\nNext step: ingest these into Qdrant vector store.")


if __name__ == "__main__":
    main()
