import socket
import numpy as np
import cv2 as cv
import time
import os

from jetbot import Robot

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
address = ('192.168.0.198', 1234)

cam_id = 0
width = 300
height = 300

camSet ='nvarguscamerasrc sensor-id=' + str(cam_id) + \
    ' ! video/x-raw(memory:NVMM), width=3264, height=2464, framerate=21/1,format=NV12 ! nvvidconv flip-method=0 ! video/x-raw, ' + \
    'width=' + str(width) + ', height=' + str(height) + ', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'

def sendUDP(frame):
    d = frame.flatten()
    s = d.tostring()
    for i in range(20):             # ((480*640*3)/20=46080) < 65535
        sock.sendto(bytes([i]) + s[i*(width*height):(i+1)*(width*height)], address)


def main():
    robot = Robot()

    pb_file = 'following/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb'
    cfg_file = 'following/faster_rcnn_resnet50_coco_2018_01_28.pbtxt'
    cvNet = cv.dnn.readNetFromTensorflow(pb_file, cfg_file)

    cap = cv.VideoCapture(camSet)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            # sendUDP(frame)
            img = frame
            rows = img.shape[0]
            cols = img.shape[1]
            cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
            cvOut = cvNet.forward()

            for detection in cvOut[0,0,:,:]:
                score = float(detection[2])
                if score > 0.3:
                    print(detection)

            # Stop the program on the ESC key
            if cv.waitKey(30) == 27:
                break
    except Exception as e:
        print(e.args[0])
    finally:
        robot.stop()
        cap.release()
        sock.close()

if __name__ == "__main__":
    main()