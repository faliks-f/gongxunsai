# encoding=utf8

import os
from tensorflow.keras.models import load_model
import argparse
import pickle
import cv2
import _thread
from imutils import paths
from utils.pipe_communicate import *


model = load_model("/home/faliks/Desktop/pythonProject/gongxunsai/outputs/seventh/modle.modle")
lb = pickle.loads(open("/home/faliks/Desktop/pythonProject/gongxunsai/outputs/seventh/lable.pickle", "rb").read())

empty_image = cv2.imread("/home/faliks/Desktop/pythonProject/gongxunsai/data/battery/img_18235.jpg")
empty_image = cv2.resize(empty_image, (1, 1))

capture = cv2.VideoCapture(0)

if not capture.isOpened():
    write('z')
    exit(1)

write('r')

predictFlag = False
image_ = None
previousLabel = None
countLabel = 0

def handleMessage(s):
    global predictFlag
    if s == 'p'.encode():
        predictFlag = True
        return
    if s == 's'.encode():
        predictFlag = False
        return


def read_image(threadName):
    global image_
    while True:
        ret, image_ = capture.read()


def send(label):
    if label == "battery":
        write('a')
        return
    if label == "bottle":
        write('b')
        return
    if label == "can":
        write('c')
        return
    if label == 'fruits':
        write('f')
        return
    if label == 'vegetable':
        write('v')
        return


try:
    _thread.start_new_thread(read_image, ("1", ))
except Exception as e:
    print("error")

while True:
    if image_ is None:
        image = None
    else:
        image = image_.copy()
    if image is None:
        print("none empty")
        cv2.imshow("empty", empty_image)
        q = cv2.waitKey(1000)
        if q == ord('q'):
            break
        if not os.path.exists(READFILE) or not os.path.exists(WRITEFILE):
            break
        continue
    output = image.copy()
    if predictFlag:
        image = cv2.resize(image, (96, 96))
        image = image.astype("float") / 255.0
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        preds = model.predict(image)
        i = preds.argmax(axis=1)[0]
        predLabel = lb.classes_[i]
        print(predLabel)
        if predLabel == previousLabel:
            countLabel += 1
        else:
            previousLabel = predLabel
            countLabel = 0
        if countLabel >= 10:
            send(predLabel)
            countLabel = 0
            predictFlag = False
    cv2.imshow("image", output)
    q = cv2.waitKey(100)
    if q == ord('q'):
        break
    if not os.path.exists(READFILE) or not os.path.exists(WRITEFILE):
        break
    handleMessage(read())

close()
# else:
#     imagePaths = sorted(list(paths.list_images(args["image"])))
#     result = {}
#     total = {}
#     for imagePath in imagePaths:
#         label = imagePath.split("/")[-2]
#         # if label == "null":
#         #     continue
#         # print(label)
#         image = cv2.imread(imagePath)
#         output = image.copy()
#         image = cv2.resize(image, (96, 96))
#         image = image.astype("float") / 255.0
#         image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#
#         preds = model.predict(image)
#         i = preds.argmax(axis=1)[0]
#         pred_label = lb.classes_[i]
#         if pred_label == label:
#             if label not in result.keys():
#                 result[label] = 0
#             else:
#                 result[label] += 1
#         if label not in total.keys():
#             total[label] = 0
#         total[label] += 1
#         # print(preds[0] * 100)
#         # text = "{}: {:.2f}%".format(pred_label, preds[0][i] * 100)
#         # cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         #
#         # cv2.imshow("Image", output)
#         # key = cv2.waitKey(0)
#         # if key == ord('q'):
#         #     break
#     print(result)
#     print(total)
