"""Reference input preparation for the exported encoder; see docs/pipeline.md."""

import argparse
from pathlib import Path

import numpy as np
import open_clip
from PIL import Image

model_name = "hf-hub:BGLab/BioTrove-CLIP"


def prepare_image(image_path, preprocess):
    # Match create_embeddings.py exactly, including RGB conversion before resizing.
    with Image.open(image_path) as image:
        tensor = preprocess(image.convert("RGB")).unsqueeze(0)
    return tensor.cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Prepare a reference Core ML input using OpenCLIP.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ios/input_image.npy"))
    args = parser.parse_args()

    # Use the existing model factory's inference transform; do not approximate it.
    # This may download the base model if it is not already cached.
    _, _, preprocess = open_clip.create_model_and_transforms(model_name)
    pixels = prepare_image(args.image, preprocess)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, pixels)
    print(f"Saved {args.output}: shape={pixels.shape}, dtype={pixels.dtype}")


if __name__ == "__main__":
    main()
