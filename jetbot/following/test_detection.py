import socket
import numpy as np
import cv2 as cv
import time

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('192.168.0.198', 1234))

width, height = 224, 224

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


# https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
# Faster-RCNN ResNet-50	2018_01_28
pb_file = 'following/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb'
cfg_file = 'following/faster_rcnn_resnet50_coco_2018_01_28.pbtxt'
cvNet = cv.dnn.readNetFromTensorflow(pb_file, cfg_file)

while True:
    try:
        img = get_frame_from_UDP()
        rows = img.shape[0]
        cols = img.shape[1]
        cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
        cvOut = cvNet.forward()

        for detection in cvOut[0,0,:,:]:
            score = float(detection[2])
            if score > 0.3:
                left = detection[3] * cols
                top = detection[4] * rows
                right = detection[5] * cols
                bottom = detection[6] * rows
                cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)

        if cv.waitKey(1) == ord('q'):
            break
    except:
        pass

    cv.imshow('img', img)
