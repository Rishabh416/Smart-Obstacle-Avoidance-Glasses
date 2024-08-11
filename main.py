# python 3.12

from transformers import pipeline
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import time
from gtts import gTTS
from playsound import playsound
import os

cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)

cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

initTime = time.time()
iterations = 0

cameraFOV = 88
focalLength = 6 # f mm 
camPixelSize = 0.00288 # d mm
camDistance = 140 # T mm

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

while True:
    ret, frame1 = cap1.read()
    ret, frame2 = cap2.read()
    # cv2.imshow("frame", frame)
    
    blurimage = cv2.GaussianBlur(frame1,(7,7),0)
    image = Image.fromarray(blurimage)
    result = pipe(image)
    depthimage = result["depth"]

    depthimagearray = np.array(depthimage)
    min1, max1, micloc1, maxloc1 = cv2.minMaxLoc(depthimagearray)
    print("image1",max1, maxloc1)

    x, y = maxloc1
    halflength = 20
    x_start = max(0, x - halflength)
    y_start = max(0, y - halflength)
    x_end = min(frame1.shape[1], x + halflength + 1)
    y_end = min(frame1.shape[0], y + halflength + 1)

    frame2blur = cv2.GaussianBlur(frame2,(7,7),0)

    gridTemplate = frame1[y_start:y_end, x_start:x_end]
    result = cv2.matchTemplate(frame2blur, gridTemplate, cv2.TM_CCOEFF_NORMED)
    min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(result)
    c, h, w = gridTemplate.shape[::-1]
    print(c, h, w)
    pixelDistance = (maxloc1[0]-(max_loc2[0] + w // 2)) # n1-n2
    objectDistance = round((focalLength/camPixelSize)*(camDistance/pixelDistance)/10) # value in cm 

    imageWidth = depthimagearray.shape[1]
    depthAngle = (maxloc1[0]*cameraFOV)/imageWidth
    text = f'closest object at {objectDistance} centimeters, {5 * round(round(depthAngle) / 5)} degrees' 
    # print(text)

    # cv2.imshow("depthimagearray", depthimagearray)
    cv2.imshow("frame2",frame2)
    cv2.imshow("template",gridTemplate)



    tts = gTTS(text=text, lang='en')
    tts.save("audio.mp3")
    playsound("audio.mp3")
    os.remove("audio.mp3") 

    iterations += 1

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

print((time.time() - initTime)/iterations)