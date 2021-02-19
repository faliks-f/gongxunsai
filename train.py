import argparse
import cv2
from common import classes
from utils import get_class
from imutils import paths

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset of images")
# ap.add_argument("-m", "--model", required=True,
#                 help="path to output trained model")
# ap.add_argument("-l", "--label-bin", required=True,
#                 help="path to output label binarizer")
# ap.add_argument("-p", "--plot", required=True,
#                 help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

imagePaths = sorted(list(paths.list_images(args["dataset"])))

data = []
labels = []

for imagePath in imagePaths:
    label = get_class.get_class(imagePath)
    if label == "null":
        continue
    print(label)
    image = cv2.imread(imagePath)
    image = image.resize(64, 64)

    data.append(image)
    labels.append(label)
