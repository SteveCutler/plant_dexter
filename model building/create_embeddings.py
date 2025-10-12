import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
import open_clip
import json
from torch import autocast
from torch.utils.data import Dataset, DataLoader
from multiprocessing import freeze_support


## Config
csv_path = "data/plantdex_clip_pairs.csv"  # image/text prompt pairs
csv_info = "data/ontario_biotrove_matches.csv" # original ontario/biotrove match csv with common names, genus, etc info.
checkpoint_path = "checkpoints/clip_finetune_step41000.pt"  # your best checkpoint
model_name = "hf-hub:BGLab/BioTrove-CLIP"
out_dir = "embeddings"
device = "mps" if torch.backends.mps.is_available() else "cpu"
batch_size = 32
num_workers = 6

#helper

def collate_skip_none(batch):
    return [s for s in batch if s]


## Dataset and Dataloader

class PlantDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        text_prompt = row["text_prompt"]

        # parse plant name from prompt
        plant_name = " ".join(text_prompt.split()[3:]).strip().lower().capitalize()

        # try to open image
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            return None  # skip broken images

        return img, text_prompt, plant_name


if __name__=="__main__":
    freeze_support()
    ## Setup

    print("making dir...")
    os.makedirs(out_dir, exist_ok=True)

    ## load model
    model, preprocess, _ = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)

    ## load weights from fine tuning

    state = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)

    model = model.to(device)
    model.eval()


    ## load the csv of image paths and text caption pairs
    df = pd.read_csv(csv_path)
    print("loaded csv pairs")
    ## load csv of info, create lookup
    info_df = pd.read_csv(csv_info)
    info_lookup = {row["scientificName"]: row for _, row in info_df.iterrows()}
    print("created name lookup")



    dataset = PlantDataset(df)
    loader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        num_workers=num_workers,
                        shuffle=False, 
                        collate_fn=collate_skip_none)

    ##set up variables

    image_embeddings, text_embeddings = [], []
    metadata = {}
    count = 1
    ## encoding loop

    #check batch
    for batch in tqdm(loader, total=len(loader)):
        if not batch:  # skip empty batch
            continue

        #unpack batch
        imgs, texts, plant_names = zip(*batch)

        #tensorize/tokenize
        imgs_t = torch.stack([preprocess(i) for i in imgs]).to(device)
        tokens = tokenizer(list(texts)).to(device)

    #use autocast for speed
        with torch.no_grad(), autocast(device_type=device, dtype=torch.float16):
            img_feats = model.encode_image(imgs_t)
            txt_feats = model.encode_text(tokens)

        #normalize
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

        #add to list
        image_embeddings.append(img_feats.cpu())
        text_embeddings.append(txt_feats.cpu())


        ## add metadata to dict (overwrite if exists already)
        for name in plant_names:
            info_row = info_lookup.get(name)
            if info_row is None:
                continue
            metadata[name] = {
            "scientificName": info_row["scientificName"],
            "common_name": info_row["common_name"],
            "family": info_row["family"],
            "genus": info_row["genus"],
            "species": info_row["species"],
            "photo_url": info_row["photo_url"]
            }

    ## SAVE ##
    print("Embeddings completed, saving...")

    image_embeddings = torch.cat(image_embeddings)
    text_embeddings = torch.cat(text_embeddings)


    #save embeddings together
    torch.save({
        "image_embeddings": image_embeddings,
        "text_embeddings": text_embeddings
    }, "embeddings/plant_dexter_embeds.pt")


    print("saved embeddings")

    ## save metadat as dict in json

    with open("embeddings/plant_dexter_metadata.json", "w") as f:
        json.dump(metadata, f, indent = 2)

    print("saved metadata")

    print("Complete : )")