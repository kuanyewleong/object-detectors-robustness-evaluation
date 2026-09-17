from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model("image.jpg")

boxes = results[0].boxes

print(boxes.xyxy)
print(boxes.conf)
print(boxes.cls)