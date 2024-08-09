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

cap1 = cv2.VideoCapture(2)
cap2 = cv2.VideoCapture(1)

initTime = time.time()
iterations = 0

cameraFOV = 88
focalLength = 6.5755 # f
camPixelSize = 0.001875 # d
camDistance =  6 # T

while True:
    ret, frame = cap1.read()
    # cv2.imshow("frame", frame)
    
    image = Image.fromarray(frame)
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    result = pipe(image)
    depthimage = result["depth"]

    depthimagearray = np.array(depthimage)
    blurimage = cv2.GaussianBlur(depthimagearray,(5,5),0)
    min, max1, micloc, maxloc1 = cv2.minMaxLoc(blurimage)
    print(max1, maxloc1)



    ret, frame = cap2.read()
    # cv2.imshow("frame", frame)
    
    image = Image.fromarray(frame)
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    result = pipe(image)
    depthimage = result["depth"]

    depthimagearray = np.array(depthimage)
    blurimage = cv2.GaussianBlur(depthimagearray,(5,5),0)
    min, max2, micloc, maxloc2 = cv2.minMaxLoc(blurimage)
    print(max2, maxloc2)



    pixelDistance = (maxloc1[0]-maxloc2[0]) # n1-n2
    objectDistance = round((focalLength/camPixelSize)*(camDistance/pixelDistance)/10) # value in cm 

    imageWidth = depthimagearray.shape[1]
    depthAngle = (maxloc1[0]*cameraFOV)/imageWidth
    text = f'closest object at {objectDistance} centimeters, {5 * round(round(depthAngle) / 5)} degrees' 
    # print(text)

    cv2.imshow("depthimagearray", depthimagearray)



    tts = gTTS(text=text, lang='en')
    tts.save("audio.mp3")
    playsound("audio.mp3")
    os.remove("audio.mp3") 

    iterations += 1

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

print((time.time() - initTime)/iterations)