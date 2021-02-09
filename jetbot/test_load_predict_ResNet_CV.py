import cv2 as cv
from tensorflow.keras.applications import resnet50

from time import time, gmtime, strftime, sleep


print("Model Load Start")
t0 = time()

model = resnet50.ResNet50()

t = gmtime(time() - t0)
print("Model Load Done in", strftime("%H:%M:%S", t))

width = 224
height = 224
cam_id = 0

# camSet ='nvarguscamerasrc sensor-id=' + str(cam_id) + \
#     ' ! video/x-raw(memory:NVMM), width=3264, height=2464, framerate=21/1,format=NV12 ! nvvidconv flip-method=0 ! video/x-raw, ' + \
#     'width=' + str(width) + ', height=' + str(height) + ', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
# cap = cv.VideoCapture(camSet)
cap = cv.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        frame = cv.resize(frame, (height, width))
        image = frame.reshape((1, 224, 224, 3))
        image = resnet50.preprocess_input(image)
        y = model.predict(image)
        print(y.argmax())

        # prob_blocked = y.argmax()
        # if prob_blocked < 0.5:
        #     robot.forward(speed)
        # else:
        #     robot.right(speed)

        sleep(0.001)

        if cv.waitKey(1)==ord('q') :
            break
except Exception as e:
    print(e.args[0])
finally:
    cap.release()