import matplotlib
matplotlib.use("Agg")
import tensorflow as tf
import argparse
import cv2
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from VGGNet.vggnet import SmallVGGNet
from VGGNet.vggnet import VGG16
from utils import get_class
from imutils import paths

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
sess = tf.compat.v1.Session(config=config)
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,
                help="path to output trained model")
ap.add_argument("-l", "--label", required=True,
                help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True,
                help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

INIT_LR = 0.0001
EPOCHS = 250
BS = 32

print("[INFO] read image...")

imagePaths = sorted(list(paths.list_images(args["dataset"])))

data = []
labels = []

for imagePath in imagePaths:
    label = imagePath.split("\\")[-2]
    if not label == "empty":
        label = "notEmpty"
    #     continue
    # if label == "pear":
    #     label = "apple"
    # if label == "null":
    #     continue
    # print(label)
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (96, 96))
    image = img_to_array(image)

    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(trainX, testX, trainY, testY) = \
    train_test_split(data, labels, test_size=0.2, random_state=42)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

print("[INFO] compiling model...")
model = SmallVGGNet.build(width=96, height=96, depth=3, classes=len(lb.classes_))
# opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
opt = Nadam(lr=INIT_LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS),
              validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
              epochs=EPOCHS)

print("[INFO] evaluating network...")
# predictions = model.predict(x=testX, batch_size=32)
# print(classification_report(testY.argmax(axis=1),
#                             predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (SmallVGGNet)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])

# save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")
model.save(args["model"], save_format="h5")
f = open(args["label"], "wb")
f.write(pickle.dumps(lb))
f.close()
