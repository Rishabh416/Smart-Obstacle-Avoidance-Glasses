# python 3.12

from transformers import pipeline
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import time
from gtts import gTTS
from playsound import playsound
import os

# setting video capture sources
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

# setting capture size to 1920x1080 (same as camera)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# time init to monitor avg loop performance
initTime = time.time()
iterations = 0

# camera parameters in millimeters
cameraFOV = 88
focalLength = 5 # f
camPixelSize = 0.001875 # d 3.6/1920
camDistance = 60 # T

# relative depth estimation model
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

while True:
    # capture image from both cameras
    ret, frame1 = cap1.read()
    ret, frame2 = cap2.read()
    
    # image pre processing, smoothing camera noise
    frame1blur = cv2.GaussianBlur(frame1,(7,7),0)
    frame2blur = cv2.GaussianBlur(frame2,(7,7),0)
    # cv2.imshow("frame1blur", frame1blur)
    # cv2.imshow("frame2blur", frame2blur)

    # process openCV image through depthAnything pipeline
    image = Image.fromarray(cv2.cvtColor(frame1blur, cv2.COLOR_BGR2RGB))
    result = pipe(image)
    depthimage = result["depth"]

    # find closest point from depth image array
    depthimagearray = np.array(depthimage)
    min1, max1, micloc1, maxloc1 = cv2.minMaxLoc(depthimagearray)
    print("image1",max1, maxloc1)
    x, y = maxloc1

    # create 40x40 image template around closest point
    halflength = 10
    x_start = max(0, x - halflength)
    y_start = max(0, y - halflength)
    x_end = min(frame1.shape[1], x + halflength + 1)
    y_end = min(frame1.shape[0], y + halflength + 1)
    gridTemplate = frame1blur[y_start:y_end, x_start:x_end]

    # roi restriction
    ymax = y + (2*halflength)
    ymin = y - (2*halflength)
    frame2ROI = frame2blur[ymin:ymax, :]

    # find matching template in image from camera 2
    result = cv2.matchTemplate(frame2ROI, gridTemplate, cv2.TM_CCOEFF) # TM_SQDIFF TM_CCOEFF
    min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(result)
    c, h, w = gridTemplate.shape[::-1]
    x_centerLoc = max_loc2[0] + w // 2
    y_centerLoc = max_loc2[1] + h // 2
    print("image2",max_val2, max_loc2)

    f = plt.figure()
    f.add_subplot(1,3, 1)
    plt.imshow(frame1blur)
    f.add_subplot(1,3, 2)
    plt.imshow(frame2blur)
    f.add_subplot(1,3, 3)
    plt.imshow(np.array(frame2ROI))
    plt.show(block=True)

    # calculate the distance of nearest object
    pixelDistance = (maxloc1[0]-x_centerLoc) # n1-n2
    objectDistance = round((focalLength/camPixelSize)*(camDistance/pixelDistance)/10) # value in cm 

    # calculation of yaw angle of closest object
    imageWidth = depthimagearray.shape[1]
    depthAngle = (maxloc1[0]*cameraFOV)/imageWidth

    text = f'closest object at {objectDistance} centimeters, {5 * round(round(depthAngle) / 5)} degrees' 
    print(text)

    # text to speech setup
    # tts = gTTS(text=text, lang='en')
    # tts.save("audio.mp3")
    # playsound("audio.mp3")
    # os.remove("audio.mp3") 

    iterations += 1

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

print((time.time() - initTime)/iterations)