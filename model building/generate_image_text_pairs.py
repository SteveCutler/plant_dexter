import os
import csv
import random

## Setup
DATA_ROOT = "/Volumes/Blue Drive/PlantDexter_v2/PlantImages"   # root folder
OUTPUT_CSV = "data/plantdex_clip_pairs.csv"

PROMPTS = [
    "a photo of {}",
    "a picture of {}",
    "an image of {}"
]

## Helper function to normalize plant species names (remove all text decoration etc)
def normalize_species_name(folder_name: str) -> str:
   
    name = folder_name.replace("_", " ").strip()
    parts = name.split()
    if parts:
        parts[0] = parts[0].capitalize()
        parts[1:] = [p.lower() for p in parts[1:]]
    return " ".join(parts)

rows = []

## process dataset
for species_folder in sorted(os.listdir(DATA_ROOT)):
    folder_path = os.path.join(DATA_ROOT, species_folder)
    if not os.path.isdir(folder_path):
        continue

    species_name = normalize_species_name(species_folder)

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(folder_path, fname)
        prompt = random.choice(PROMPTS).format(species_name)
        rows.append([image_path, prompt])

## output csv file
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "text_prompt"])
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")
