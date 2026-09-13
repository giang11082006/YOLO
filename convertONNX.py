from ultralytics import YOLO

model = YOLO("runs/detect/train-4/weights/best.pt")

# convert model yolo sang onnx
# model.export(format="onnx", imgsz=640)

model2 = YOLO("runs/detect/train-4/weights/best.onnx")

model2.predict("aabc.mp4",
               save=True,
               conf=0.25)
