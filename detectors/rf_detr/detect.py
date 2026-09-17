from rfdetr import RFDETRNano

model = RFDETRNano()
detections = model.predict("image.jpg", threshold=0.25)

print(detections.xyxy)
print(detections.confidence)
print(detections.class_id)