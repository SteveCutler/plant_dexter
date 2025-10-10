import os, math, argparse, json, random
from dataclasses import asdict, dataclass

import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import pandas as pd
import open_clip

## Config

csv_path: str = "data/plantdex_clip_pairs.csv"
model_name: str = "hf-hub:BGLab/BioTrove-CLIP"
batch_size: int = 16
epochs: int = 2
lr: float = 5e-6
save_every: int = 500            
out_dir: str = "checkpoints"
seed: int = 1337  
device = "mps" if torch.backends.mps.is_available() else "cpu" # use gpu if possible, else cpu

## Seed
torch.manual_seed(seed)
random.seed(seed)


## Model and tokenizer

model, preprocess, _ = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)
model = model.to(device)
model.train()


## Resume from checkpoint if available
# resume_path = "checkpoints/clip_finetune_step6000.pt"
# if os.path.exists(resume_path):
#     ckpt = torch.load(resume_path, map_location=device)
#     if "state_dict" in ckpt:
#         model.load_state_dict(ckpt["state_dict"])
#         print(f"Resumed full checkpoint weights from {resume_path}")
#     else:
#         model.load_state_dict(ckpt)
#         print(f"Resumed model weights from {resume_path}")
# else:
#     print("Starting fresh training run")

## Dataset
class PairsCSV(Dataset):
    def __init__(self, csv_path, preprocess, tokenizer):
        self.df = pd.read_csv(csv_path)
        self.preprocess = preprocess
        self.tokenizer = tokenizer
    
    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except:
            img = Image.new("RGB", (224, 224), (0,0,0))

        txt = self.tokenizer(row["text_prompt"])

        return self.preprocess(img), txt.squeeze(0)

ds = PairsCSV(csv_path, preprocess, tokenizer)
dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

## Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

## Helper functions

# create out directory if doesn't exist
os.makedirs(out_dir, exist_ok=True)

glob_step = 0

## Main loop
for epoch in range(epochs):

    running_loss = 0.0

    for i, (images, texts) in enumerate(tqdm(dl)):

        if isinstance(texts, list):
            texts = torch.stack(texts)

        images, texts = images.to(device), texts.to(device)

    ## using autocast for M2 casting bug
    ##with torch.autocast(device_type="cpu"):
        #encode image and text features - do image on mps and text on cpu for speed
        
        # Encode image and text features directly on MPS
        img_feat = model.encode_image(images)
        txt_feat = model.encode_text(texts)

        # Normalize and compute logits
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        ##print(img_feat.dtype, txt_feat.dtype, img_feat.device, txt_feat.device)

        #cast tensors to f32 to fix compatibility (cpu tensor is bfloat16)
        img_feat = img_feat.to(dtype=torch.float32, device=torch.device(device))
        txt_feat = txt_feat.to(dtype=torch.float32, device=torch.device(device))


        ## print(img_feat.dtype, txt_feat.dtype, img_feat.device, txt_feat.device)

        #compare image and text features in matrix
        logit_scale = model.logit_scale.exp()
        logits = logit_scale * img_feat @ txt_feat.t()

        #in case batch size isn't full 16 on final step - using images.size(0)
        targets = torch.arange(images.size(0), device=device)

        #compute loss
        loss = (
            torch.nn.functional.cross_entropy(logits, targets)
            + torch.nn.functional.cross_entropy(logits.T,targets)
            )/ 2
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        glob_step += 1

        # every 500 steps save checkpoint
        if glob_step % save_every == 0:
            ckpt_path = f"{out_dir}/clip_finetune_step{glob_step}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Saved checkpoint → {ckpt_path}")

    # epoch summary

    ## average loss across epochs
    epoch_loss = running_loss / len(dl)
    print(f"Epoch {epoch+1} complete. Avg loss: {epoch_loss:.4f}")
    torch.save(model.state_dict(), f"{out_dir}/clip_finetune_epoch{epoch+1}.pt")



final_path = f"{out_dir}/clip_finetune_final.pt"
torch.save(model.state_dict(), final_path)
print("Training complete, model saved to {final_path}")



