import cv2 as cv
import numpy as np
import socket

import tensorflow as tf
from tensorflow.keras.applications import resnet50

from time import time, gmtime, strftime, sleep
import os

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


print("Model Load Start")
t0 = time()

trained_model_path = "new_trained_from_resnet50_256"

model = tf.keras.models.load_model(trained_model_path)

t = gmtime(time() - t0)
print("Model Load Done in", strftime("%H:%M:%S", t))


speed = 0.1
left_alpha = 0.95
right_alpha = 0.90

robot = Robot()

robot.left_motor.alpha = left_alpha
robot.right_motor.alpha = right_alpha


while cap.isOpened():
    try:
        ret, frame = cap.read()
        # sendUDP(frame)

        image = frame.reshape(-1, 224, 224, 3)
        x = resnet50.preprocess_input(image)
        y = model.predict(x)
        print(y, y.argmax())

        # prob_blocked = y[0][0]
        # if prob_blocked < 0.5:
        #     robot.right(speed)
        # else:
        #     robot.forward(speed)
        # sleep(0.001)

        # if cv.waitKey(1)==ord('q') :
        #     break
    except Exception as e:
        print(e.args[0])

cap.release()
robot.stop()
sock.close()

