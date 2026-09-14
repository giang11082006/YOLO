import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = MODEL_PATH = "runs/detect/train-4/weights/best.onnx"
IMAGE_PATH = "a.jpg"

session = ort.InferenceSession(MODEL_PATH,
                               providers= ["CPUExecutionProvider"])

# READ IMAGE
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Không đọc được ảnh: {IMAGE_PATH}")

original_height, original_width = image.shape[:2]
print(image.shape) # (416,416,3)

image_resized = cv2.resize(image, (640,640))
print(image_resized.shape)  #(640, 640, 3)

# BGR -> RGB
image_rgb = cv2.cvtColor(
    image_resized,
    cv2.COLOR_BGR2RGB
)

#preprocessing
image_float = image_rgb.astype(np.float32) / 255.0 

# HWC -> CHW
input_tensor = np.transpose(image_float,    # (3,640,640)
                            (2,0,1))

print(input_tensor.shape)

# thêm batch dimension
input_tensor = np.expand_dims(input_tensor, axis=0) # (1,3,640,640)

print(input_tensor.shape)
model_inputs = session.get_inputs()
correct_input_name = model_inputs[0].name

# run
outputs = session.run(
    None,
    {correct_input_name: input_tensor}   
)


'''
NMS sẽ:

Bỏ box có confidence thấp.
Chọn box có confidence cao nhất.
So sánh nó với các box còn lại.
Nếu hai box chồng lên nhau quá nhiều → bỏ box thấp hơn.
Lặp lại.
'''
  # list
output = outputs[0]  #(1,6,8400)

raw_output = output[0]   #(6,8400)

boxes = raw_output[:4].T # (8400,4)
classes_scores = raw_output[4:].T # (8400,2)

# class and conf
class_ids = np.argmax(classes_scores, axis = 1)
confidences = np.max(classes_scores, axis = 1) # (8400,)
# conf filter 
CONF_THRESHOLD = 0.25

mask = confidences >= CONF_THRESHOLD

boxes = boxes[mask]
confidences = confidences[mask]
class_ids = class_ids[mask]


# boxes  : [x_center, y_center, width, height]
# opencv NMS  want boxes : [x_top_left, y_top_left, width, height]

boxes_xywh = []
for box in boxes:
    x_cen, y_cen, w, h = box
    x_new = x_cen - w / 2
    y_new = y_cen - h / 2

    boxes_xywh.append([int(x_new),
                       int(y_new),
                       int(w),
                       int(h)])
# NMS BANG OPENCV
indices = cv2.dnn.NMSBoxes(         # [ 2 15  7 22]
    boxes_xywh,
    confidences.tolist(),
    0.25, #score threshold
    0.45 #  NMS IoU threshold
)

print(f"Số box sau NMS: {len(indices)}")

# Model trả tọa độ theo ảnh 640x640, nên quy đổi về ảnh gốc trước khi vẽ.
scale_x = original_width / 640
scale_y = original_height / 640

for i in indices:

    i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i

    x, y, w, h = boxes_xywh[i]
    x = int(x * scale_x)
    y = int(y * scale_y)
    w = int(w * scale_x)
    h = int(h * scale_y)

    confidence = confidences[i]

    class_id = class_ids[i]

    label = f"{class_id}: {confidence:.2f}"

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        label,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )

output_path = "inference_result.jpg"
cv2.imwrite(output_path, image)
print(f"Đã lưu ảnh kết quả: {output_path}")