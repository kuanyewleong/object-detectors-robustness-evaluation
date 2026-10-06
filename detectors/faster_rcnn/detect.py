import torch
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights
)

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]

IMAGE_PATH = Path(r"dataset\spoon\spoon2_0%_topdown_white_plain.jpg")
OUTPUT_PATH = Path("baseline_test") / f"{IMAGE_PATH.stem}_detected.jpg"
CONFIDENCE_THRESHOLD = 0.4
CAPTION_FONT_SIZE = 32

# TorchVision uses "remote" for the remote control class.
TARGET_CLASSES = {"spoon": "spoon", "book": "book", "remote": "remote control"}
COLORS = {"spoon": "red", "book": "blue", "remote": "green"}

model = fasterrcnn_resnet50_fpn_v2(weights=weights)
model.eval()

image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = weights.transforms()(image)

with torch.inference_mode():
    results = model([input_tensor])

boxes = results[0]["boxes"]
conf = results[0]["scores"]
cls = results[0]["labels"]

draw = ImageDraw.Draw(image)
try:
    caption_font = ImageFont.truetype("arial.ttf", CAPTION_FONT_SIZE)
except OSError:
    caption_font = ImageFont.load_default(size=CAPTION_FONT_SIZE)
detection_count = 0

for box, score, label in zip(boxes, conf, cls):
    class_name = categories[label.item()]
    confidence = score.item()
    if class_name not in TARGET_CLASSES or confidence < CONFIDENCE_THRESHOLD:
        continue

    display_name = TARGET_CLASSES[class_name]
    caption = f"{display_name}: {confidence:.3f}"
    print(caption)
    detection_count += 1

    x1, y1, x2, y2 = box.tolist()
    color = COLORS[class_name]
    draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    text_bounds = draw.textbbox((0, 0), caption, font=caption_font)
    text_width = text_bounds[2] - text_bounds[0]
    text_height = text_bounds[3] - text_bounds[1]
    padding = 4
    caption_x = max(0, min(x1, image.width - text_width - 2 * padding))
    caption_y = max(0, y1 - text_height - 2 * padding)
    draw.rectangle(
        (caption_x, caption_y,
         caption_x + text_width + 2 * padding,
         caption_y + text_height + 2 * padding),
        fill=color,
    )
    text_position = (
        caption_x + padding - text_bounds[0],
        caption_y + padding - text_bounds[1],
    )
    draw.text(text_position, caption, font=caption_font, fill="white")

if detection_count == 0:
    print("No spoon, book, or remote control detected above the confidence threshold.")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
image.save(OUTPUT_PATH)
print(f"Saved annotated image to: {OUTPUT_PATH}")
