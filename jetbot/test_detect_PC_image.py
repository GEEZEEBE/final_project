import cv2 as cv
import numpy as np

import tensorflow as tf
from object_detection.utils import label_map_util

import time
import os
import socket


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

width, height = 224, 224


PATH_TO_MODEL_DIR = "exported-models/my_model"
PATH_TO_LABELS = ""
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

image_path = 'data/c90ead02-6689-11eb-b09e-7ac0a4e907bd.jpg'

try:
    image = cv.imread(image_path)
    rows, cols, channels = image.shape

    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)

    scores = detections['detection_scores'][0]
    boxes = detections['detection_boxes'][0]
    class_ids = detections['detection_classes'][0]
    for class_id, score, box in zip(class_ids, scores, boxes):
        if score > 0.5:
            left = box[0] * cols
            top = box[1] * rows
            right = box[2] * cols
            bottom = box[3] * rows
            cv.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255))
    cv.imshow('Jetbot', image)

    cv.waitKey()

except Exception as e:
    print(e.args[0])




