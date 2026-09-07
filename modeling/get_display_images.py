
import os
import json
import requests
import time
from tqdm import tqdm
from PIL import Image
from io import BytesIO

# --- CONFIG ---
input_path  = "embeddings/plant_dexter_metadata_wiki.json"
output_path = "embeddings/plant_dexter_metadata_with_local_images.json"
image_dir   = "data/plant_images"

os.makedirs(image_dir, exist_ok=True)
HEADERS = {"User-Agent": "PlantDexter/1.0 (contact: steve@example.com)"}

# --- Load metadata ---
with open(input_path, "r") as f:
    metadata = json.load(f)
print(f"Loaded {len(metadata)} plant entries")

# -----------------------------------------------------------
# Wikimedia Commons helpers
# -----------------------------------------------------------

def fetch_candidate_images_wikimedia(scientific_name, retries=2):
    title = scientific_name.replace(" ", "_")
    url = (
        f"https://commons.wikimedia.org/w/api.php?"
        f"action=query&titles={title}&prop=images&format=json"
    )
    for attempt in range(1, retries + 1):
        try:
            res = requests.get(url, headers=HEADERS, timeout=10)
            if res.status_code == 200:
                data = res.json()
                pages = data.get("query", {}).get("pages", {})
                if not pages:
                    return []
                images = list(next(iter(pages.values())).get("images", []))
                return [
                    img["title"] for img in images
                    if img["title"].lower().endswith((".jpg", ".jpeg", ".png"))
                ][:2]
        except Exception:
            time.sleep(1.0 * attempt)
    return []


def fetch_and_save_wikimedia_image(file_title, out_path):
    info_url = (
        f"https://commons.wikimedia.org/w/api.php?"
        f"action=query&titles={file_title.replace(' ', '_')}&prop=imageinfo&iiprop=url&format=json"
    )
    try:
        res = requests.get(info_url, headers=HEADERS, timeout=10)
        data = res.json()
        pages = data.get("query", {}).get("pages", {})
        for p in pages.values():
            if "imageinfo" in p:
                img_url = p["imageinfo"][0]["url"]
                img_res = requests.get(img_url, headers=HEADERS, timeout=10)
                if img_res.status_code == 200:
                    img = Image.open(BytesIO(img_res.content))
                    img = img.convert("RGB")
                    img.thumbnail((512, 512))
                    img.save(out_path, format="JPEG", quality=70)
                    return True
    except Exception as e:
        print(f"  ⚠️  Wikimedia download failed: {e}")
    return False


# -----------------------------------------------------------
# iNaturalist fallback helpers
# -----------------------------------------------------------

def fetch_inaturalist_images(scientific_name):
    """Return up to 2 iNaturalist photo URLs for a species."""
    try:
        query_url = (
            f"https://api.inaturalist.org/v1/taxa?q={scientific_name}&per_page=1"
        )
        res = requests.get(query_url, headers=HEADERS, timeout=10)
        if res.status_code == 200:
            data = res.json()
            results = data.get("results", [])
            if not results:
                return []
            taxon = results[0]
            photos = taxon.get("taxon_photos", [])
            urls = []
            for p in photos:
                url = p.get("photo", {}).get("medium_url")
                if url:
                    urls.append(url)
                if len(urls) >= 2:
                    break
            return urls
    except Exception as e:
        print(f"  ⚠️  iNaturalist fetch failed: {e}")
    return []


def fetch_and_save_inat_image(img_url, out_path):
    try:
        res = requests.get(img_url, headers=HEADERS, timeout=10)
        if res.status_code == 200:
            img = Image.open(BytesIO(res.content))
            img = img.convert("RGB")
            img.thumbnail((512, 512))
            img.save(out_path, format="JPEG", quality=70)
            return True
    except Exception as e:
        print(f"  ⚠️  iNat download failed: {e}")
    return False


# -----------------------------------------------------------
# Main loop
# -----------------------------------------------------------

missing = 0
for sci_name, info in tqdm(metadata.items(), desc="Downloading images"):
    safe_name = sci_name.replace(" ", "_")
    plant_folder = os.path.join(image_dir, safe_name)
    os.makedirs(plant_folder, exist_ok=True)

    if "local_images" in info and info["local_images"]:
        continue

    local_paths = []

    # --- 1️⃣ Try Wikimedia first
    wiki_candidates = fetch_candidate_images_wikimedia(sci_name)
    for i, title in enumerate(wiki_candidates):
        out_path = os.path.join(plant_folder, f"wikimedia_{i+1}.jpg")
        if fetch_and_save_wikimedia_image(title, out_path):
            local_paths.append(out_path)

    # --- 2️⃣ If none found, try iNaturalist fallback
    if not local_paths:
        inat_urls = fetch_inaturalist_images(sci_name)
        for i, url in enumerate(inat_urls):
            out_path = os.path.join(plant_folder, f"inaturalist_{i+1}.jpg")
            if fetch_and_save_inat_image(url, out_path):
                local_paths.append(out_path)

    info["local_images"] = local_paths
    if not local_paths:
        missing += 1

    time.sleep(0.3)

# --- Save updated metadata ---
with open(output_path, "w") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"✅ Done! Images saved under '{image_dir}'")
print(f"❌ Missing images for {missing} / {len(metadata)} species.")
