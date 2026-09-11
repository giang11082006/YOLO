from ultralytics import YOLO

model = YOLO("yolo26n.pt") # pre_trained model

model.train(
    data="dataset/data.yaml",
    epochs=10,
    imgsz=640,
    batch=16
)
# Detect trên video
# results = model2.predict(
#     source="11964619_2160_3840_30fps.mp4",
#     save=True,
#     conf=0.25,

# )

print("done")