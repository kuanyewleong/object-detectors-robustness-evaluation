from pathlib import Path
from typing import Any, cast

import cv2
from PIL import ImageColor
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

IMAGE_PATH = Path(r"dataset\book\Book2_0%_diagonal_grey_plain.jpg")
OUTPUT_PATH = Path("baseline_test") / f"{IMAGE_PATH.stem}_yolo26_detected.jpg"
CONFIDENCE_THRESHOLD = 0.4

# Ultralytics uses "remote" for the remote control class.
TARGET_CLASSES = {"spoon": "spoon", "book": "book", "remote": "remote control"}
COLORS = {"spoon": "red", "book": "blue", "remote": "green"}

model = YOLO("yolo26x.pt")
target_class_ids = [
    class_id for class_id, name in model.names.items() if name in TARGET_CLASSES
]
results = model.predict(
    source=str(IMAGE_PATH),
    classes=target_class_ids,
    conf=CONFIDENCE_THRESHOLD,
    verbose=False,
)
result = cast(Any, next(iter(results)))

result.names = {
    class_id: TARGET_CLASSES.get(name, name)
    for class_id, name in cast(dict[int, str], result.names).items()
}

annotator = Annotator(result.orig_img.copy(), line_width=3)

for box in result.boxes:
    class_id = int(box.cls.item())
    class_name = result.names[class_id]
    confidence = box.conf.item()
    caption = f"{class_name}: {confidence:.3f}"
    print(caption)

    color_name = COLORS[model.names[class_id]]
    # Ultralytics' OpenCV annotator expects BGR rather than RGB.
    color_bgr = ImageColor.getrgb(color_name)[::-1]
    annotator.box_label(box.xyxy[0].tolist(), caption, color=color_bgr)

if len(result.boxes) == 0:
    print("No spoon, book, or remote control detected above the confidence threshold.")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
annotated_image = cast(Any, annotator.result())
if not cv2.imwrite(str(OUTPUT_PATH), annotated_image):
    raise OSError(f"Could not save annotated image to: {OUTPUT_PATH}")
print(f"Saved annotated image to: {OUTPUT_PATH}")
