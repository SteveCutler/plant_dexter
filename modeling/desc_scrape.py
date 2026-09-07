import json
import requests
import time
import re
from tqdm import tqdm

# --- CONFIG ---
input_path  = "embeddings/plant_dexter_metadata.json"
output_path = "embeddings/plant_dexter_metadata_wiki.json"

HEADERS = {
    "User-Agent": "PlantDexter/1.0 (contact: steve@example.com)"
}

# --- Load existing metadata ---
with open(input_path, "r") as f:
    metadata = json.load(f)

print(f"Loaded {len(metadata)} plant entries")

# --- Helper: clean html ---
def clean_html(raw_html):
    return re.sub(r'<.*?>', '', raw_html or '')

# --- Helper: fetch Wikipedia summary + image ---
def fetch_wiki_data(scientific_name):
    title = scientific_name.replace(" ", "_")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    try:
        res = requests.get(url, headers=HEADERS, timeout=5)
        if res.status_code != 200:
            return None, None
        data = res.json()

        desc = data.get("extract") or data.get("description") or data.get("extract_html")
        desc = clean_html(desc)
       

        return desc
    except Exception as e:
        print(f"Error fetching {scientific_name}: {e}")
        return None, None

# --- Iterate and enrich ---
for name, info in tqdm(metadata.items(), desc="Scraping Wikipedia"):
    # remove old photo_url key if present
    info.pop("photo_url", None)

    # skip if description already exists
    if info.get("description"):
        continue

    desc = fetch_wiki_data(info["scientificName"])
    info["description"] = desc
    

    time.sleep(0.2)  # polite delay (5 requests/sec)

# --- Save enriched metadata ---
with open(output_path, "w") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"✅ Done! Saved enriched metadata to {output_path}")
