# Phat hien nguoi khong doi mu bao hiem voi YOLO

Du an custom train **YOLO26n** de nhan dien nguoi co doi mu va khong doi mu bao hiem. Model sau khi huan luyen duoc dung de predict video `aabc.mp4`, tao ra video da gan bounding box va nhan cho tung doi tuong.

## Ket qua video

## Demo

[▶️ Xem video demo](https://drive.google.com/drive/folders/1Lx3WEsT451KcRCbu3BpvsCQjtSx6Zlxi?usp=sharing)
- Model tot nhat: [`runs/detect/train-4/weights/best.pt`](runs/detect/train-4/weights/best.pt)

> Luu y: GitHub co the khong phat truc tiep dinh dang `.avi`. Tai file video dau ra ve de xem ket qua day du.

## Nhan du lieu

| ID | Nhan | Y nghia |
| --- | --- | --- |
| 0 | `With Helmet` | Nguoi co doi mu bao hiem |
| 1 | `Without Helmet` | Nguoi khong doi mu bao hiem |

Du lieu nam trong thu muc `dataset/`, duoc chia thanh `train`, `valid` va `test`. Dataset Bike-Helmet co 629 anh va duoc export tu Roboflow theo dinh dang YOLO. Thong tin chi tiet ve nguon va giay phep CC BY 4.0 nam tai [`dataset/README.dataset.txt`](dataset/README.dataset.txt).

## Cau truc du an

```text
.
|-- dataset/
|   |-- train/                 # Anh va nhan huan luyen
|   |-- valid/                 # Anh va nhan validation
|   |-- test/                  # Anh va nhan kiem thu
|   `-- data.yaml              # Duong dan dataset va danh sach nhan
|-- runs/detect/train-4/
|   |-- weights/best.pt        # Checkpoint PyTorch tot nhat
|   `-- weights/best.onnx      # Model ONNX dung cho inference anh
|   `-- results.png            # Bieu do qua trinh huan luyen
|-- aabc.mp4                   # Video dau vao
|-- a.jpg                      # Anh dau vao cho ONNX inference
|-- train.py                   # Script custom train
|-- predict.py                 # Script predict video
`-- inference.py               # Script inference anh bang ONNX Runtime
```

## Cai dat

Yeu cau Python 3.10+.

```bash
pip install ultralytics
```

De chay inference ONNX tren mot anh, cai them cac thu vien:

```bash
pip install opencv-python numpy onnxruntime
```

Kiem tra cai dat:

```bash
yolo checks
```

## Huan luyen model

Script [`train.py`](train.py) khoi tao model pretrained `yolo26n.pt` va fine-tune tren `dataset/data.yaml` voi cac tham so:

- `epochs=10`
- `imgsz=640`
- `batch=16`

Chay train:

```bash
python train.py
```

Ket qua cua moi lan train duoc Ultralytics luu trong `runs/detect/`. Checkpoint can dung cho predict thuong la `weights/best.pt`.

## Predict video aabc

[`predict.py`](predict.py) da duoc cau hinh de dung checkpoint tot nhat va predict video `aabc.mp4` voi nguong confidence `0.25`.

```bash
python predict.py
```

Video da gan nhan duoc luu tu dong vao thu muc `runs/detect/predict-*`. Ket qua hien tai la [`runs/detect/predict-9/aabc.avi`](runs/detect/predict-9/aabc.avi).

De predict mot video khac, sua bien `source` trong `predict.py`:

```python
model.predict(
    source="duong_dan_den_video.mp4",
    save=True,
    conf=0.25,
)
```

## Inference anh bang ONNX

[`inference.py`](inference.py) chay model ONNX bang ONNX Runtime, sau do ve bounding box va nhan lop len mot anh.

Mac dinh script su dung:

- Model: `runs/detect/train-4/weights/best.onnx`
- Anh dau vao: `a.jpg`
- Kich thuoc input model: `640x640`
- Confidence threshold: `0.25`
- NMS IoU threshold: `0.45`
- Anh ket qua: `inference_result.jpg`

Chay lenh:

```bash
python inference.py
```

Quy trinh inference gom cac buoc: doc anh, resize ve `640x640`, chuyen BGR sang RGB, chuyen du lieu sang tensor `NCHW`, chay model ONNX, loc confidence, ap dung NMS va ve ket qua len anh goc. Toa do bounding box duoc quy doi tu anh `640x640` ve kich thuoc anh ban dau truoc khi luu.

Anh dau vao va model co the duoc thay doi bang cach sua hai bien `IMAGE_PATH` va `MODEL_PATH` trong [`inference.py`](inference.py).

## Ket qua huan luyen hien tai

Lan train `train-4` chay 10 epochs tren anh kich thuoc 640. Tai epoch cuoi, ket qua tren tap validation la:

| Precision | Recall | mAP@50 | mAP@50-95 |
| --- | --- | --- | --- |
| 0.7675 | 0.8221 | 0.8057 | 0.4788 |

Xem bieu do loss va metric tai [`runs/detect/train-4/results.png`](runs/detect/train-4/results.png), va confusion matrix tai [`runs/detect/train-4/confusion_matrix.png`](runs/detect/train-4/confusion_matrix.png).

## Luu y

- Hieu qua phat hien phu thuoc vao goc quay, anh sang, kich thuoc nguoi trong khung hinh va chat luong video.
- Neu muon cai thien ket qua cho boi canh thuc te, nen bo sung anh/video cung dieu kien camera, gan nhan lai va fine-tune them model.
