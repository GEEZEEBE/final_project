import cv2 as cv
import numpy as np

import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from time import time, gmtime, strftime, sleep
import os
import socket


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

width, height = 224, 224


print("Model Load Start")

t0 = time()

trained_model_path = "new_trained_from_mobilenet_modi"

model = tf.keras.models.load_model(trained_model_path)

t = gmtime(time() - t0)
print("Model Load Done in", strftime("%H:%M:%S", t))

filename = 'dataset/validation/blocked/1a30d816-6538-11eb-9520-ccd9ac0a6757.jpg'
filename = 'dataset/validation/free/0fce7040-65bf-11eb-81d0-16f63a1aa8c9.jpg'

try:
    image = cv.imread(filename)
    # image = load_img(filename, target_size=(width, height))
    x = img_to_array(image)
    x = x.reshape(-1, 224, 224, 3)
    x = mobilenet_v2.preprocess_input(x)
    y = model.predict(x)
    print(y, y.argmax())

    prob_blocked = y[0][0]
    if prob_blocked > 0.5:
        cv.putText(image, "Blocked", (10, 20), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv.putText(image, "Free", (10, 20), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
    cv.imshow("jetbot", image)

    cv.waitKey()

except Exception as e:
    print(e.args[0])



