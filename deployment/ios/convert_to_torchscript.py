import argparse
import json
from pathlib import Path

import torch
import open_clip

model_name = "hf-hub:BGLab/BioTrove-CLIP"
checkpoint_path = "checkpoints/clip_finetune_step41000.pt"
output_path = "artifacts/ios/plantdex_image_encoder_traced.pt"


class ImageEncoder(torch.nn.Module):
    """Accept an OpenCLIP-preprocessed NCHW tensor and return normalized features."""

    def __init__(self, visual_model):
        super().__init__()
        self.visual = visual_model

    def forward(self, x):
        feats = self.visual(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats


def load_finetuned_weights(model, checkpoint):
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # Strict loading prevents an incomplete or incompatible checkpoint being exported.
    model.load_state_dict(state, strict=True)


def main():
    parser = argparse.ArgumentParser(description="Export the fine-tuned BioTrove image encoder.")
    parser.add_argument("--checkpoint", default=checkpoint_path)
    parser.add_argument("--output", type=Path, default=Path(output_path))
    args = parser.parse_args()

    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    model.to("cpu")
    load_finetuned_weights(model, args.checkpoint)
    model.eval()
    encoder = ImageEncoder(model.visual).eval()

    dummy = torch.randn(1, 3, 224, 224)
    # Retain the prototype's trace workaround. Real-image output parity is still unverified.
    traced_model = torch.jit.trace(encoder, dummy, strict=False, check_trace=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    traced_model.save(str(args.output))

    # These transforms remain outside the graph and must be reproduced by the app.
    preprocessing = {
        "model_name": model_name,
        "checkpoint": str(args.checkpoint),
        "input_name": "input_image",
        "input_shape": [1, 3, 224, 224],
        "input_dtype": "float32",
        "preprocess_cfg": open_clip.get_model_preprocess_cfg(model),
        "transform": repr(preprocess),
        "torch_version": str(torch.__version__),
        "open_clip_version": open_clip.__version__,
    }
    manifest = args.output.with_suffix(".preprocessing.json")
    manifest.write_text(json.dumps(preprocessing, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} and {manifest}")


if __name__ == "__main__":
    main()
