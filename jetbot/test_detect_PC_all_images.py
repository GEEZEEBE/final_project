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

image_path = 'track_data'
images = [f for f in os.listdir(image_path)]


def detection_center(bbox):
    """Computes the center x, y coordinates of the object"""
    center_x = (bbox[0] + bbox[2]) / 2.0 - 0.5
    center_y = (bbox[1] + bbox[3]) / 2.0 - 0.5
    return (center_x, center_y)

def norm(vec):
    """Computes the length of the 2D vector"""
    return np.sqrt(vec[0]**2 + vec[1]**2)

def closest_detection(detections):
    """Finds the detection closest to the image center"""
    closest_detection = None
    for det in detections:
        center = detection_center(det)
        if closest_detection is None:
            closest_detection = det
        elif norm(detection_center(det)) < norm(detection_center(closest_detection)):
            closest_detection = det
    return closest_detection

i = 0
vstack = []
for img in images:
    try:
        image = cv.imread(image_path + '/' + img)
        rows, cols, channels = image.shape

        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = detect_fn(input_tensor)

        scores = detections['detection_scores'][0]
        boxes = detections['detection_boxes'][0]

        matching_detections = [box for score, box in zip(scores, boxes) if score > 0.5]

        if len(matching_detections) != 0:

            # for box in matching_detections:
            #     left = box[0] * cols
            #     top = box[1] * rows
            #     right = box[2] * cols
            #     bottom = box[3] * rows
            #     cv.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255))

            box = closest_detection(matching_detections)
            print(box)

            left = box[0] * cols
            top = box[1] * rows
            right = box[2] * cols
            bottom = box[3] * rows
            cv.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255))

        i += 1
        if i % 10 == 1:
            hstack = image.copy()
        elif i % 10 > 1:
            hstack = np.hstack((hstack, image))
        elif i % 10 == 0:
            vstack.append(hstack)

        if i == 40:
            break

    except Exception as e:
        print(e.args[0])

cv.imshow('Track', np.vstack((vstack[0], vstack[1], vstack[2], vstack[3])))
cv.waitKey()
