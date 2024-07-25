# python 3.12

from transformers import pipeline
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import time

cap = cv2.VideoCapture(0)

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
    plt.imshow(depthimagearray)
    plt.show()

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break