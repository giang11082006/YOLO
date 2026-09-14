import onnxruntime as ort
import numpy as np 

MODEL_PATH = "runs/detect/train-4/weights/best.onnx"

#1 load model onnx

session = ort.InferenceSession(MODEL_PATH,
                               providers=["CPUExecutionProvider"])

#2. thoong tin input

inputs = session.get_inputs()   # list

for i , inp in enumerate(inputs):
    print(f"\nInput {i}")
    print("Name: ", inp.name)
    print("Shape: ", inp.shape)
    print("type: ", inp.type)

'''
Input 0
Name:  images
Shape:  [1, 3, 640, 640]
type:  tensor(float)
'''

outputs = session.get_outputs()
for i, oup in enumerate(outputs):
    print(f"\nOUTput {i}")
    print("Name: ", oup.name)
    print("Shape: ", oup.shape)
    print("type: ", oup.type)

'''
OUTput 0
Name:  output0
Shape:  [1, 6, 8400]
type:  tensor(float)'''

print("\n" + "=" * 60)
print("MODEL INFO")
print("=" * 60)

print("Inputs :", len(inputs))
print("Outputs:", len(outputs))

print("\nProviders:")
print(session.get_providers())
