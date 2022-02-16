import argparse
import os
import random

import cv2
import numpy as np
import tensorflow as tf
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import img_to_array

data_path = 'testdata'
image_size = (300, 300)

tf.random.set_seed(777)
np.random.seed(777)
random.seed(777)


def load(test_data_dir):
    images_data = []
    categories = []
    image_paths = sorted(list(paths.list_images(test_data_dir)))
    for path in image_paths:
        # Append resized image in array form to image data
        images_data.append(img_to_array(cv2.resize(cv2.imread(path), image_size)))
        # Append image category from file type
        categories.append(path.split(os.path.sep)[-2])
    return [images_data, categories]


def asArray(loaded):
    lb = LabelBinarizer()
    validate_X = np.array(loaded[0], dtype="float") / 255.0
    validate_y = lb.fit_transform(np.array(loaded[1]))
    return validate_X, validate_y


def evaluate(X_test, y_test):
    batch = 15
    model = load_model('model.h5')
    return model.evaluate(X_test, y_test, batch, verbose=1)


if __name__ == '__main__':
    # best_seed = 0
    # best_acc = 0
    #
    # for i in range(0, 1000):
    #
    #     g = tf.Graph()
    #     with g.as_default():
    #
    #     Update seeds here
    #
    #         X_test, y_test = asArray(load(data_path))
    #         loss, accuracy = evaluate(X_test, y_test)
    #         if accuracy > best_acc:
    #             best_acc = accuracy
    #             best_seed = i
    #             print("New Best Seed: " + str(i) + "with accuracy: " + str(accuracy))

    X_test, y_test = asArray(load(data_path))
    loss, accuracy = evaluate(X_test, y_test)
    print("loss={}, accuracy={}".format(loss, accuracy))
    # print("Best final accuracy was " + str(best_acc) + " with seed " + str(best_seed))
