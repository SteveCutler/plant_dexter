import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

input_path = "artifacts/ios/plantdex_image_encoder_traced.pt"
output_path = "artifacts/ios/PlantDex_ImageEncoder_FP16.mlpackage"


def main():
    parser = argparse.ArgumentParser(description="Convert the traced image encoder to Core ML.")
    parser.add_argument("--input", type=Path, default=Path(input_path))
    parser.add_argument("--output", type=Path, default=Path(output_path))
    args = parser.parse_args()

    model_ts = torch.jit.load(str(args.input), map_location="cpu")
    model_ts.eval()

    # The app supplies a normalized RGB tensor, not raw image pixels.
    # See preprocess_image.py and docs/pipeline.md for the OpenCLIP input transform.
    mlmodel = ct.convert(
        model_ts,
        inputs=[ct.TensorType(name="input_image", shape=(1, 3, 224, 224), dtype=np.float32)],
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16,
    )
    mlmodel.input_description["input_image"] = (
        "Float32 NCHW RGB tensor using BioTrove/OpenCLIP inference preprocessing; "
        "see the traced model's .preprocessing.json sidecar."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(args.output))
    print(f"Saved {args.output}; bundle its preprocessing specification with the model.")


if __name__ == "__main__":
    main()
