import torch
from PIL import Image
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights
)

weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT

model = ssdlite320_mobilenet_v3_large(weights=weights)
model.eval()

image = Image.open("image.jpg").convert("RGB")
image = weights.transforms()(image)

with torch.inference_mode():
    results = model([image])

boxes = results[0]["boxes"]
conf = results[0]["scores"]
cls = results[0]["labels"]

print(boxes)
print(conf)
print(cls)