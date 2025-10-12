import torch
import numpy as np
import os

## config
input_path = "/Volumes/Blue Drive/PlantDexter_v2/data/embeddings/plant_dexter_embeds.pt" 
output_path = "/Volumes/Blue Drive/PlantDexter_v2/data/embeddings/plant_dexter_embeds_fp16.pt" 

## load
if input_path.endswith(".pt"):
    emb = torch.load(input_path, map_location="cpu")
    if isinstance(emb, dict):
       
        for k, v in emb.items():
            if isinstance(v, torch.Tensor):
                emb[k] = v.half()
        torch.save(emb, output_path)
    else:
        emb = emb.half()
        torch.save(emb, output_path)


torch.save(emb, output_path, _use_new_zipfile_serialization=True)


