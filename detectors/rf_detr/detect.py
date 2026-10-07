from pathlib import Path

import cv2
from PIL import ImageColor
from rfdetr import RFDETRSmall
from rfdetr.assets.coco_classes import COCO_CLASSES

IMAGE_PATH = Path(r"dataset\book\Book1_0%_diagonal_grey_plain.jpg")
OUTPUT_PATH = Path("baseline_test") / f"{IMAGE_PATH.stem}_rf_detr_detected.jpg"
CONFIDENCE_THRESHOLD = 0.4

# COCO names the remote control class "remote".
TARGET_CLASSES = {"spoon": "spoon", "book": "book", "remote": "remote control"}
COLORS = {"spoon": "red", "book": "blue", "remote": "green"}

image = cv2.imread(str(IMAGE_PATH))
if image is None:
    raise FileNotFoundError(f"Could not read input image: {IMAGE_PATH}")

model = RFDETRSmall()
# RF-DETR expects RGB arrays; keep the BGR original for drawing and saving.
detections = model.predict(
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB), threshold=CONFIDENCE_THRESHOLD
)
detection_count = 0

boxes = getattr(detections, "xyxy", None)
confidences = getattr(detections, "confidence", None)
class_ids = getattr(detections, "class_id", None)

if boxes is None or confidences is None or class_ids is None:
    raise TypeError(
        "RF-DETR returned an unexpected result without bounding boxes, "
        "confidence scores, or class IDs."
    )

for box, confidence, class_id in zip(boxes, confidences, class_ids):
    class_name = COCO_CLASSES.get(int(class_id))
    if class_name not in TARGET_CLASSES or confidence < CONFIDENCE_THRESHOLD:
        continue

    caption = f"{TARGET_CLASSES[class_name]}: {float(confidence):.3f}"
    print(caption)
    detection_count += 1

    x1, y1, x2, y2 = (int(value) for value in box)
    # Convert the named RGB color to OpenCV's BGR channel order.
    color = ImageColor.getrgb(COLORS[class_name])[::-1]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=3)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        caption, font, font_scale, thickness
    )
    padding = 4
    caption_x = max(0, min(x1, image.shape[1] - text_width - 2 * padding))
    caption_y = max(0, y1 - text_height - baseline - 2 * padding)
    cv2.rectangle(
        image,
        (caption_x, caption_y),
        (caption_x + text_width + 2 * padding,
         caption_y + text_height + baseline + 2 * padding),
        color,
        thickness=-1,
    )
    cv2.putText(
        image, caption,
        (caption_x + padding, caption_y + text_height + padding),
        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )

if detection_count == 0:
    print("No spoon, book, or remote control detected above the confidence threshold.")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
if not cv2.imwrite(str(OUTPUT_PATH), image):
    raise OSError(f"Could not save annotated image to: {OUTPUT_PATH}")
print(f"Saved annotated image to: {OUTPUT_PATH}")
