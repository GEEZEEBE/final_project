from jetbot import Robot
import cv2 as cv

import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.saved_model import tag_constants
import numpy as np

from time import time, gmtime, strftime

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

t0 = time()

input_saved_model = "trt_mobilenet_0"

print("Model Load Start")

saved_model_loaded = tf.saved_model.load(input_saved_model, tags=[tag_constants.SERVING])
signature_keys = list(saved_model_loaded.signatures.keys())
print(signature_keys)

infer = saved_model_loaded.signatures['serving_default']
print(infer.structured_outputs)

t = gmtime(time() - t0)
print("Model Load Done in", strftime("%H:%M:%S", t))


speed = 0.1
left_alpha = 0.95
right_alpha = 0.90

robot = Robot()

robot.left_motor.alpha = left_alpha
robot.right_motor.alpha = right_alpha

width = 224
height = 224
cam_id = 0

camSet ='nvarguscamerasrc sensor-id=' + str(cam_id) + \
    ' ! video/x-raw(memory:NVMM), width=3264, height=2464, framerate=21/1,format=NV12 ! nvvidconv flip-method=0 ! video/x-raw, ' + \
    'width=' + str(width) + ', height=' + str(height) + ', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cap = cv.VideoCapture(camSet)

try:
    while True:
        ret, frame = cap.read()
        x = cv.resize(frame, (height, width))
        x = np.expand_dims(x, axis=0)
        x = mobilenet_v2.preprocess_input(x)
        x = tf.constant(x)
        labeling = infer(x)
        preds = labeling['predictions'].numpy()
        print(preds)

        # prob_blocked = y.argmax()
        # if prob_blocked < 0.5:
        #     robot.forward(speed)
        # else:
        #     robot.right(speed)

        time.sleep(0.001)

        if cv.waitKey(1)==ord('q') :
            break
except Exception as e:
    print(e.args[0])
finally:
    cap.release()
    robot.stop()