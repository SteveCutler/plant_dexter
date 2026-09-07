from datasets import load_dataset
import pandas as pd
from collections import Counter
import csv

## Load Ontario native species list
ontario_native = pd.read_csv("data/ontario_native_filtered.csv")

def normalize_name(n):
    n = str(n).lower().replace("_", " ").strip()
    parts = n.split()
    return " ".join(parts[:2]) 

ontario_names = set(ontario_native["SCIENTIFIC_NAME"].apply(normalize_name))

## Load biotrove dataset
biotrove = load_dataset("BGLab/Biotrove", split="train", streaming=True)

counts = Counter()

## Prepare CSV writer for matched rows
out_file = open("data/ontario_biotrove_matches.csv", "w", newline="", encoding="utf-8")
writer = csv.DictWriter(out_file, fieldnames=[
    "photo_id", "scientificName", "kingdom", "phylum", "class", "order",
    "family", "genus", "species", "common_name", "taxonRank", "photo_url"
])
writer.writeheader()

## iterate through data rows
for i, row in enumerate(biotrove):
    if row.get("kingdom", "").lower() != "plantae":
        continue

    name = normalize_name(row.get("scientificName", ""))
    if name in ontario_names:
        counts[name] += 1
        writer.writerow(row) 

## list matches every 100000 images
    if i % 100000 == 0 and i > 0:
        print(f"Processed {i:,} rows — matched {len(counts)} species so far...")

out_file.close()

## save matches
species_counts = pd.DataFrame(counts.items(), columns=["SCIENTIFIC_NAME", "image_count"])
ontario_native = ontario_native.merge(species_counts, on="SCIENTIFIC_NAME", how="left").fillna({"image_count": 0})
ontario_native.to_csv("data/ontario_biotrove_overlap.csv", index=False)

print("Done")

