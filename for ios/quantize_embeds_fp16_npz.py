import torch
import numpy as np

#paths
input_path = "/Volumes/Blue Drive/PlantDexter_v2/data/embeddings/plant_dexter_embeds_fp16.pt"
output_path = "/Volumes/Blue Drive/PlantDexter_v2/data/embeddings/plant_dexter_embeds_fp16_2.npz"

#load 
print(f"Loading {input_path} ...")
data = torch.load(input_path, map_location="cpu")

#convert
np_data = {}
for k, v in data.items():
    if isinstance(v, torch.Tensor):
        np_data[k] = v.cpu().numpy().astype(np.float16)
        print(f"{k:25s} -> {v.shape}, dtype {v.dtype}")
    else:
        np_data[k] = np.array(v)

#save
np.savez_compressed(output_path, **np_data)
print(f"\nSaved compressed FP16 embeddings to: {output_path}")
