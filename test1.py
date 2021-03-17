import os
from tensorflow.keras.models import load_model
import argparse
import pickle
import cv2
import _thread
from imutils import paths

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model = load_model("./outputs/eleven/model.model")
lb = pickle.loads(open("./outputs/eleven/label.pickle", "rb").read())
print(lb.classes_)
capture = cv2.VideoCapture(1)

while True:
    ret, image = capture.read()
    if image is None:
        print("none")
    output = image.copy()
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    preds = model.predict(image)
    i = preds.argmax(axis=1)[0]
    predLabel = lb.classes_[i]
    print(predLabel)
    cv2.imshow("image", output)
    cv2.waitKey(1)