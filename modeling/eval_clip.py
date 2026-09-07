import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import open_clip

# -----------------------------
# Config
# -----------------------------
csv_path = "data/val_pairs.csv"      # your held-out validation CSV
model_name = "hf-hub:BGLab/BioTrove-CLIP"
checkpoint_path = "checkpoints/clip_finetune_step41000.pt"  # adjust as needed
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load model + tokenizer
# -----------------------------
print(f"🚀 Loading model on {device}")
# Use the same inference transform as reference embeddings and deployment.
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)
model = model.to(device)
model.eval()

# Load fine-tuned weights
if checkpoint_path and os.path.exists(checkpoint_path):
    print(f"🔁 Loading fine-tuned weights from {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    # Handle either full checkpoint dict or plain state_dict
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
else:
    print("⚠️ No fine-tuned checkpoint found, evaluating base model.")

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(csv_path)
print(f"🗂️  Evaluating {len(df)} image-text pairs")

image_embeds, text_embeds = [], []

with torch.no_grad():
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Load and preprocess image
        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        # Tokenize text
        text_tokens = tokenizer([row["text_prompt"]]).to(device)

        # Encode
        img_feat = model.encode_image(img_tensor)
        txt_feat = model.encode_text(text_tokens)

        # Normalize
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)

        image_embeds.append(img_feat.cpu())
        text_embeds.append(txt_feat.cpu())

# Stack all embeddings
image_embeds = torch.cat(image_embeds, dim=0)
text_embeds = torch.cat(text_embeds, dim=0)

# -----------------------------
# Compute similarity matrix
# -----------------------------
print("🔍 Computing cosine similarity...")
similarity = image_embeds @ text_embeds.T  # [N x N]
ranks = similarity.argsort(dim=-1, descending=True)

# -----------------------------
# Compute metrics
# -----------------------------
top1, top5 = 0, 0
n = len(df)
for i in range(n):
    ranking = ranks[i].tolist()
    if i == ranking[0]:
        top1 += 1
    if i in ranking[:5]:
        top5 += 1

top1_acc = top1 / n * 100
top5_acc = top5 / n * 100

print(f"✅ Top-1 Accuracy: {top1_acc:.2f}%")
print(f"✅ Top-5 Accuracy: {top5_acc:.2f}%")
print("🎯 Evaluation complete.")
