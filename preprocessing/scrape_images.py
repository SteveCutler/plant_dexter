import os
import requests
import pandas as pd
import time
from urllib.parse import quote

## config
DATA_ROOT = os.environ.get("PLANTDEXTER_IMAGE_ROOT", "data/PlantImages")
CSV_PATH = "data/low_count_species.csv" 
TARGET_COUNT = 100               
MAX_PER_SPECIES = 500            
SLEEP_BETWEEN_REQUESTS = 0.5     

## load low count species list
species_df = pd.read_csv(CSV_PATH)
print(f" Loaded {len(species_df)} species to scrape")

## helper functions

def normalize_species_name(raw_name: str) -> str:
    """Convert names like 'salix_daphnoides' → 'Salix daphnoides'."""
    name = raw_name.strip().replace("_", " ")
    parts = name.split()
    # Capitalize genus, keep species/subspecies lowercase
    if len(parts) >= 1:
        parts[0] = parts[0].capitalize()
    if len(parts) >= 2:
        parts[1:] = [p.lower() for p in parts[1:]]
    return " ".join(parts)


def fetch_inat_photos(scientific_name, per_page=200):
    """Query the iNaturalist API for a given species name."""
    photos = []
    page = 1
    base_url = "https://api.inaturalist.org/v1/observations"
    query = quote(scientific_name)
    while len(photos) < MAX_PER_SPECIES and page < 10:
        url = (
                    f"{base_url}?taxon_name={query}&photos=true"
                    f"&quality_grade=research&per_page={per_page}&page={page}"
                )

        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                print(f"Error fetching {scientific_name} page {page}: {resp.status_code}")
                break
            results = resp.json().get("results", [])
            if not results:
                break
            for r in results:
                for photo in r.get("photos", []):
                    pid = str(photo.get("id"))
                    purl = photo.get("url")
                    if not pid or not purl:
                        continue

                    # Get a decent medium-sized image URL
                    purl = purl.replace("square.", "medium.").replace("small.", "medium.")
                    photos.append((pid, purl))
            page += 1
            time.sleep(0.2)
        except Exception as e:
            print(f"Exception fetching {scientific_name}: {e}")
            break
    return photos

def download_image(url, dest_path):
    """Download an image from URL to a destination path."""
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with open(dest_path, "wb") as f:
                f.write(r.content)
            return True
    except Exception as e:
        print("Download error:", e)
    return False


## main loop
for _, row in species_df.iterrows():
    species = row["species"] if "species" in row else row[0]
    folder = os.path.join(DATA_ROOT, species)
    os.makedirs(folder, exist_ok=True)

    # Count how many images we already have
    existing_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    n_existing = len(existing_files)
    if n_existing >= TARGET_COUNT:
        print(f"{species}: already has {n_existing} images")
        continue

    needed = TARGET_COUNT - n_existing

    ## normalize species name
    species_name = normalize_species_name(species)
    print(f"\n{species_name}: {n_existing} images, need {needed} more...")

    photos = fetch_inat_photos(species_name)
    if not photos:
        print("No photos found on iNaturalist")
        continue

    added = 0
    for pid, url in photos:
        filename = os.path.join(folder, f"{pid}.jpg")
        if os.path.exists(filename):  # skip duplicates
            continue
        if download_image(url, filename):
            added += 1
        if added >= needed:
            break

    print(f"Added {added} new images for {species}")
    time.sleep(SLEEP_BETWEEN_REQUESTS)

