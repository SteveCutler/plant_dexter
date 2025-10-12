import torch
import numpy as np

input_path = "/Volumes/Blue Drive/PlantDexter_v2/data/embeddings/plant_dexter_embeds.pt"
output_path = "/Volumes/Blue Drive/PlantDexter_v2/data/embeddings/plant_dexter_embeds_int8.pt"

data = torch.load(input_path, map_location="cpu")

quantized = {}

## loop finds the float vectors
for key, tensor in data.items():
    if isinstance(tensor, torch.Tensor) and tensor.dtype in (torch.float16, torch.float32):
        # find max val
        max_val = tensor.abs().max().item()
        #calculate scale factor for quantization
        scale = max_val / 127.0 if max_val > 0 else 1.0
        #normalizes, rounds down and clamps in int8 bounds
        q_tensor = torch.clamp((tensor / scale).round(), -127, 127).to(torch.int8)
        #stores quantized tensor and scale value in dict for dequantizatino
        quantized[key] = {"values": q_tensor, "scale": scale}
    else:
        quantized[key] = tensor

torch.save(quantized, output_path)

