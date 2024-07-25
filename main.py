# python 3.12

from transformers import pipeline
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import time

cap = cv2.VideoCapture(0)

initTime = time.time()
iterations = 0

cameraFOV = 80

while True:
    ret, frame = cap.read()
    # cv2.imshow("frame", frame)
    
    image = Image.fromarray(frame)
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    result = pipe(image)
    depthimage = result["depth"]

    depthimagearray = np.array(depthimage)
    blurimage = cv2.GaussianBlur(depthimagearray,(5,5),0)
    min, max, micloc, maxloc = cv2.minMaxLoc(blurimage)

    imageWidth = depthimagearray.shape[1]
    depthAngle = (maxloc[0]*cameraFOV)/imageWidth
    print(f'closest object at {5 * round(round(depthAngle) / 5)} degrees')

    print(maxloc)
    cv2.imshow("depthimagearray", depthimagearray)
    iterations += 1

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

print((time.time() - initTime)/iterations)