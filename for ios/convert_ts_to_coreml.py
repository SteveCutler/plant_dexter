import coremltools as ct
import torch



## load traced model and set to eval
model_ts = torch.jit.load("/Volumes/Blue Drive/PlantDexter_v2/data/convert_for_ios/plantdex_image_encoder_traced.pt")
model_ts.eval()

## dummy tensor
dummy = torch.randn(1, 3, 224, 224)

mlmodel = ct.convert(
    model_ts,
    inputs=[ct.ImageType(name="input_image", shape=dummy.shape)],
    compute_units=ct.ComputeUnit.ALL, 
    convert_to="mlprogram",           
    minimum_deployment_target=ct.target.iOS16,
    compute_precision=ct.precision.FLOAT16,
)

#save
mlmodel.save("PlantDex_ImageEncoder_FP16.mlpackage")

