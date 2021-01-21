from jetbot import Robot
import cv2 as cv

import tensorflow as tf
from tensorflow.keras.applications import resnet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import time

trained_model_path = "new_trained_from_resnet50"

model = tf.saved_model.load(trained_model_path)

speed = 0.1
left_alpha = 0.95
right_alpha = 0.90

robot = Robot()

robot.left_motor.alpha = left_alpha
robot.right_motor.alpha = right_alpha

width = 224
height = 224

cap = cv.VideoCapture(0)
try:
    while True:
        ret, frame = cap.read()
        image = load_img(frame, target_size=(width, height))
        image = img_to_array(image)
        image = image.reshape((1, 224, 224, 3))
        image = resnet50.preprocess_input(image)
        y = model.predict(image)
        print(y)

        # prob_blocked = y[0]
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