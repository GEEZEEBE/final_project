import socket
import numpy as np
import cv2 as cv

sock1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock1.bind(('192.168.0.198', 1234))
sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock2.bind(('192.168.0.198', 4321))

width = 224
height = 224

s1 = [b'\xff'*(width*height) for x in range(20)]
s2 = [b'\xff'*(width*height) for x in range(20)]
while True:
    try:
        picture = b''
        data1, addr1 = sock1.recvfrom((width*height)+1)
        data2, addr2 = sock2.recvfrom((width*height)+1)
        s1[data1[0]] = data1[1:(width*height)+1]
        s2[data2[0]] = data2[1:(width*height)+1]
        if data1[0] == 19:
            for i in range(20):
                picture += s1[i]
            frame1 = np.fromstring(picture, dtype=np.uint8)
            frame1 = frame1.reshape(height, width, 3)
            # cv.imshow('Jetbot1', frame1)
        if data2[0] == 19:
            for i in range(20):
                picture += s2[i]
            frame2 = np.fromstring(picture, dtype=np.uint8)
            frame2 = frame2.reshape(height, width, 3)
            # cv.imshow('Jetbot2', frame2)

        stack = np.hstack((frame1, frame2))
        cv.imshow("Jetbot", stack)

        if cv.waitKey(1) == ord('q'):
            break
    except:
        pass

sock1.close()
sock2.close()
