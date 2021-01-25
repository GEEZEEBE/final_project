import socket
import numpy
import cv2 as cv

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('192.168.0.198', 1234))

width = 224
height = 224

s = [b'\xff'*(width*height) for x in range(20)]
while True:
    picture = b''
    data, addr = sock.recvfrom((width*height)+1)
    s[data[0]] = data[1:(width*height)+1]
    if data[0] == 19:
        for i in range(20):
            picture += s[i]
        frame = numpy.fromstring(picture, dtype=numpy.uint8)
        frame = frame.reshape(height, width, 3)
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
