import socket
import cv2 as cv

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

cam_id = 0
width = 224
height = 224

camSet ='nvarguscamerasrc sensor-id=' + str(cam_id) + \
    ' ! video/x-raw(memory:NVMM), width=3264, height=2464, framerate=21/1,format=NV12 ! nvvidconv flip-method=0 ! video/x-raw, ' + \
    'width=' + str(width) + ', height=' + str(height) + ', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cap = cv.VideoCapture(camSet)

while cap.isOpened():
    ret, frame = cap.read()         # frame (480, 640, 3)
    d = frame.flatten()
    s = d.tostring()
    for i in range(20):             # ((480*640*3)/20=46080) < 65535
        sock.sendto(bytes([i]) + s[i*(width*height):(i+1)*(width*height)], ('192.168.0.198', 1234))
cap.release()