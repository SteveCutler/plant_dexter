import torch
import open_clip

## setup

model_name = "hf-hub:BGLab/BioTrove-CLIP"
checkpoint_path = "checkpoints/clip_finetune_step41000.pt"  
device = "cpu"

## load model and preprocess
model, preprocess, _ = open_clip.create_model_and_transforms(model_name)
model.to(device)

## load fine tuned weights
state = torch.load(checkpoint_path, map_location=device)
if "state_dict" in state:
    state = state["state_dict"]

# put in eval mode
model.eval()

## grab the visual encoder from the model
class ImageEncoder(torch.nn.Module):
    def __init__(self, visual_model):
        super().__init__()
        self.visual = visual_model

    def forward(self, x):
        feats = self.visual(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

encoder = ImageEncoder(model.visual)

##define tensor shape for trace
dummy = torch.randn(1, 3, 224, 224)

## trace and save model

traced_model = torch.jit.trace(encoder, dummy, strict=False, check_trace=False) ## check trace false because it was throwing a crazy error
traced_model.save("plantdex_image_encoder_traced.pt")
