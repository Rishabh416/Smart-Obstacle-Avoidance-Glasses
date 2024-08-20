import numpy as np
import cv2
import matplotlib.pyplot as plt


cap1 = cv2.VideoCapture(0)
# cap2 = cv2.VideoCapture(1)

cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    ret, frame1 = cap1.read()
    # ret, frame2 = cap2.read()

    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    frame1 = cv2.GaussianBlur(frame1,(5,5),0)
    # frame2 = cv2.GaussianBlur(frame2,(5,5),0)

    frame1 = cv2.adaptiveThreshold(frame1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,5)
    # frame2 = cv2.adaptiveThreshold(frame2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

    frame1 = edges = cv2.Canny(frame1,100,200)
    # frame2 = edges = cv2.Canny(frame2,100,300)

    f = plt.figure()
    f.add_subplot(1,1, 1)
    plt.imshow(np.array(frame1))
    # f.add_subplot(1,2, 2)
    # plt.imshow(np.array(frame2))
    plt.show(block=True)