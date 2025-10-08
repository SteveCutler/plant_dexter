import os
import pandas as pd
from collections import Counter


DATA_ROOT = "/Volumes/Blue Drive/PlantDexter_v2/PlantImages"  
THRESHOLD = 100             

def count_images(root):
    counts = Counter()
    for dirpath, _, filenames in os.walk(root):
        if dirpath == root:
            continue
        species = os.path.basename(dirpath)

        ## count image files only
        n = sum(
            1 for f in filenames
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        )
        counts[species] = n
    return counts

if __name__ == "__main__":
    counts = count_images(DATA_ROOT)

    df = pd.DataFrame(counts.items(), columns=["species", "image_count"])
    df = df.sort_values("image_count")

    ## Save full counts for reference
    ##df.to_csv("data/species_image_counts.csv", index=False)

    ## Filter for low-count species
    low_df = df[df["image_count"] < THRESHOLD]
    low_df.to_csv("data/low_count_species.csv", index=False)

    print(f"✅ Total species: {len(df)}")
    print(f"⚠️  Species below {THRESHOLD} images: {len(low_df)}")
    low_df.head(n=21)

