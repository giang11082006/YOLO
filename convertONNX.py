from ultralytics import YOLO

model = YOLO("runs/detect/train-4/weights/best.pt")

# convert model yolo sang onnx
model.export(format="onnx", imgsz=640)

