import os
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import hashlib
import time

## controls
CSV_PATH = "ontario_biotrove_matches.csv"
OUTPUT_DIR = "/Volumes/Blue Drive/PlantDexter_v2/PlantImages"
MAX_IMAGES_PER_SPECIES = 200
MAX_THREADS = 16  
RETRY_LIMIT = 3

## load biotrove matched data
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df):,} image records")

## normalize species name for searching and folder matching
def clean_name(n):
    return n.lower().replace(" ", "_").replace("/", "_").strip()

df["species_dir"] = df["scientificName"].apply(clean_name)

## create folders
os.makedirs(OUTPUT_DIR, exist_ok=True)
species_groups = df.groupby("species_dir")

## helper function for downloading a single image
def download_image(species_dir, url, idx):
    folder = os.path.join(OUTPUT_DIR, species_dir)
    os.makedirs(folder, exist_ok=True)

    ## create filename based on url hash
    filename = hashlib.md5(url.encode()).hexdigest() + ".jpg"
    path = os.path.join(folder, filename)

    ## Skip if already exists
    if os.path.exists(path):
        return True

    ## attempt to download image
    for attempt in range(RETRY_LIMIT):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200 and r.content:
                with open(path, "wb") as f:
                    f.write(r.content)
                return True
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return False

## download loop
total_images = 0
with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    futures = []
    for species_dir, group in species_groups:
        subset = group.head(MAX_IMAGES_PER_SPECIES)
        for i, row in subset.iterrows():
            futures.append(executor.submit(download_image, species_dir, row["photo_url"], i))

    ## Progress bar
    for f in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
        total_images += 1


