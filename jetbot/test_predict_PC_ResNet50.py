import cv2 as cv
import numpy as np

import tensorflow as tf
from tensorflow.keras.applications import resnet50

from time import time, gmtime, strftime, sleep
import os
import socket


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

width, height = 224, 224

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('192.168.0.198', 1234))

def get_frame_from_UDP():
    s = [b'\xff'*(width*height) for x in range(20)]
    while True:
        picture = b''
        data, addr = sock.recvfrom((width*height)+1)
        s[data[0]] = data[1:(width*height)+1]
        if data[0] == 19:
            for i in range(20):
                picture += s[i]
            frame = np.fromstring(picture, dtype=np.uint8)
            try:
                frame = frame.reshape(height, width, 3)
            except:
                pass
            break
    return frame



print("Model Load Start")

t0 = time()

trained_model_path = "new_trained_from_resnet50_jun1"

model = tf.keras.models.load_model(trained_model_path)

t = gmtime(time() - t0)
print("Model Load Done in", strftime("%H:%M:%S", t))

while True:
    try:
        frame = get_frame_from_UDP()

        image = frame.reshape(-1, 224, 224, 3)
        x = resnet50.preprocess_input(image)
        y = model.predict(x)
        print(y, y.argmax())

        if len(y) == 1:
            prob_blocked = y[0][0]
            if prob_blocked < 0.5:
                cv.putText(frame, "Blocked", (10, 20), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
            else:
                cv.putText(frame, "Free", (10, 20), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
            cv.imshow("jetbot", frame)

        if cv.waitKey(1)==ord('q') :
            break
    except Exception as e:
        print(e.args[0])

sock.close()


