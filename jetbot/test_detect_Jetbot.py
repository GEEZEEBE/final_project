import cv2 as cv
import numpy as np

import tensorflow as tf
from object_detection.utils import label_map_util

import time
import os
import socket

from jetbot import Robot

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
address = ('192.168.0.198', 1234)

def sendUDP(frame):
    d = frame.flatten()
    s = d.tostring()
    for i in range(20):             # ((480*640*3)/20=46080) < 65535
        sock.sendto(bytes([i]) + s[i*(width*height):(i+1)*(width*height)], address)


width, height = 224, 224
cam_id = 0

camSet ='nvarguscamerasrc sensor-id=' + str(cam_id) + \
    ' ! video/x-raw(memory:NVMM), width=1028, height=720, framerate=21/1, format=NV12 ! nvvidconv flip-method=0 ! video/x-raw, ' + \
    'width=' + str(width) + ', height=' + str(height) + ', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cap = cv.VideoCapture(camSet)



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


speed = 0.15
left_alpha = 0.98
right_alpha = 0.90

robot = Robot()

robot.left_motor.alpha = left_alpha
robot.right_motor.alpha = right_alpha


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


while cap.isOpened():
    try:
        ret, frame = cap.read()
        sendUDP(frame)
        rows, cols, channels = frame.shape

        input_tensor = tf.convert_to_tensor(frame)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = detect_fn(input_tensor)

        scores = detections['detection_scores'][0]
        boxes = detections['detection_boxes'][0]

        matching_detections = [box for score, box in zip(scores, boxes) if score > 0.5]

        box = closest_detection(matching_detections)
        print("box : ", box)

        if box is None:
            robot.forward(speed)
            print("speed : ", speed)
        else:
            center = detection_center(box)
            center = np.float16(center)[0]
            print("center : ", center)
            robot.set_motors(speed + 0.8 * center, speed - 0.8 * center)
            print("left : ", speed + 0.8 * center, "right : ", speed - 0.8 * center)

    except Exception as e:
        print(e.args[0])

cap.release()
robot.stop()
sock.close()


