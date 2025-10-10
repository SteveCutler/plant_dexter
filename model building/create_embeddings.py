import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
import open_clip
import pandas as pd
import numpy as np
import os
import copy

## Setup
CSV_PATH = "data/plantdex_clip_pairs.csv"
MODEL_NAME = "hf-hub:BGLab/BioTrove-CLIP" ## using open clip biotrove model
BATCH_SIZE = 32
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu" ## set to run on metal by default

## load model
model_full, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(MODEL_NAME)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

# Create two copies: one on GPU/MPS, one on CPU
model_img = copy.deepcopy(model_full).to(DEVICE)
model_txt = copy.deepcopy(model_full).to("cpu")

model_img.eval()
model_txt.eval()


## load the csv of image paths and text caption pairs
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} image-text pairs from {CSV_PATH}")

image_embeddings = []
text_embeddings = []
labels = []

## embedding loop
for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Embedding batches"):
    batch = df.iloc[i:i + BATCH_SIZE]

    # Load and preprocess images
    images = []
    for path in batch["image_path"]:
        try:
            img = preprocess_val(Image.open(path).convert("RGB"))
            images.append(img)
        except Exception as e:
            print(f"⚠️ Skipping unreadable image: {path} ({e})")

    if not images:
        continue

    images = torch.stack(images).to(DEVICE)
    texts = tokenizer(batch["text_prompt"].tolist())

    with torch.no_grad():
        # Run image encoder on GPU/MPS
        img_emb = model_img.encode_image(images)

        # Run text encoder on CPU (MPS has a known bug)
        
        txt_emb = model_txt.encode_text(texts)
        

        # Normalize embeddings for cosine similarity
        img_emb /= img_emb.norm(dim=-1, keepdim=True)
        txt_emb /= txt_emb.norm(dim=-1, keepdim=True)

        image_embeddings.append(img_emb.cpu())
        text_embeddings.append(txt_emb.cpu())
        labels.extend(batch["text_prompt"].tolist())

## save output
image_embeddings = torch.cat(image_embeddings).numpy()
text_embeddings = torch.cat(text_embeddings).numpy()

np.save("embeddings/image_embeddings.npy", image_embeddings)
np.save("embeddings/text_embeddings.npy", text_embeddings)
np.save("embeddings/text_labels.npy", np.array(labels))

print(f"Total embedded pairs: {len(labels)}")
