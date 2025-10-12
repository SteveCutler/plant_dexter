import torch
from PIL import Image
import open_clip
from torchvision import transforms
import numpy as np
import os

## SETUP
model_name = "hf-hub:BGLab/BioTrove-CLIP"
device = "mps" if torch.backends.mps.is_available() else "cpu"

model, preprocess, _ = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)

# Load fine tuned weights
model.load_state_dict(torch.load("checkpoints/clip_finetune_step41000.pt", map_location=device))
model = model.to(device)
model.eval()

## test images
images = [
    "tests/cappadocian_navelwort.jpg",
]

## labels
labels = [
    "Onoclea sensibilis",        # Sensitive fern (definitely in your training set)
    "Omphalodes cappadocica",    # Cappadocian navelwort
    "Brunnera macrophylla",     # false forget me not
    "Myosotis sylvatica", # forget me not
    "Athyrium filix-femina",     # Lady fern
    "Omphalodes verna",          # Maidenhair fern
    "Anchusa azurea",       # Bracken fern
    "Cystopteris bulbifera",     # Bulblet fern
    "Lobelia erinus",        # Cinnamon fern
    "Nemophila menziesii"        # Interrupted fern
]

## encode
with torch.no_grad():
    # Encode text once
    text_tokens = tokenizer(labels)
    text_features = model.encode_text(text_tokens.to(device))
    text_features /= text_features.norm(dim=-1, keepdim=True)

    for img_path in images:
        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarities
        sims = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        top_prob, top_label = sims[0].topk(3)

        print(f"\n🪴 {os.path.basename(img_path)}")
        for i in range(3):
            print(f"  {labels[top_label[i]]}: {top_prob[i].item():.3f}")
