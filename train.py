import matplotlib

matplotlib.use("Agg")
import argparse
import cv2
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from VGGNet.vggnet import SmallVGGNet
from VGGNet.vggnet import VGG16
from utils import get_class
from imutils import paths

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,
                help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True,
                help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True,
                help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

INIT_LR = 0.001
EPOCHS = 75
BS = 1

print("[INFO] read image...")

imagePaths = sorted(list(paths.list_images(args["dataset"])))

data = []
labels = []

for imagePath in imagePaths:
    label = get_class.get_class(imagePath)
    if label == "null":
        continue
    # print(label)
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    # image = image / 255.0
    # image = np.expand_dims(image, 0)

    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float32")
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(trainX, testX, trainY, testY) = \
    train_test_split(data, labels, test_size=0.2, random_state=42)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

print("[INFO] compiling model...")
model = VGG16.build(width=64, height=64, depth=3, classes=len(lb.classes_))
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS),
              validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
              epochs=EPOCHS)

print("[INFO] evaluating network...")
predictions = model.predict(x=testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=lb.classes_))

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
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()
