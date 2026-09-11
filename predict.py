from ultralytics import YOLO

model = YOLO("runs/detect/train-4/weights/best.pt")

model.predict(
    source="aabc.mp4",
    save=True,
    conf=0.25
)