from ultralytics import YOLO

model = YOLO("yolo26n.pt") # pre_trained model

model.train(
    data="dataset/data.yaml",
    epochs=10,
    imgsz=640,
    batch=16
)
print("done")