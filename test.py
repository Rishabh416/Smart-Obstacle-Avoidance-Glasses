from transformers import pipeline
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

cap1 = cv2.VideoCapture(0) # left
cap2 = cv2.VideoCapture(1) # right

cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

pixelDiff = 175

while True:
    ret, frame1 = cap1.read()
    ret, frame2 = cap2.read()

    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    frame1 = cv2.GaussianBlur(frame1,(5,5),0)
    frame2 = cv2.GaussianBlur(frame2,(5,5),0)

    frame2 = frame2[:,pixelDiff:1920]
    frame1 = frame1[:,0:1920-pixelDiff]

    image = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    result = pipe(image)
    depthimage = result["depth"]

    depthimagearray = np.array(depthimage)
    min1, max1, micloc1, maxloc1 = cv2.minMaxLoc(depthimagearray)
    print("image1",max1, maxloc1)
    x, y = maxloc1

    halflength = 100
    x_start = max(0, x - halflength)
    y_start = max(0, y - halflength)
    x_end = min(frame1.shape[1], x + halflength + 1)
    y_end = min(frame1.shape[0], y + halflength + 1)
    gridTemplate = frame1[y_start:y_end, x_start:x_end]

    ymax = y + (2*halflength)
    ymin = y - (2*halflength)

    result = cv2.matchTemplate(frame2[ymin:ymax, :], gridTemplate, cv2.TM_CCOEFF) # TM_SQDIFF TM_CCOEFF
    min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(result)
    print("image2",max_val2, max_loc2)
    c, h, w = gridTemplate.shape[::-1]
    x_centerLoc = max_loc2[0] + w // 2

    pixelDistance = (maxloc1[0]-x_centerLoc)
    objectDistance = round((0.0004*(pixelDistance**2))+(0.3767*(pixelDistance))+110.8)

    f = plt.figure()
    f.add_subplot(1,3, 1)
    plt.imshow(np.array(frame1))
    f.add_subplot(1,3, 2)
    plt.imshow(np.array(gridTemplate))
    f.add_subplot(1,3, 3)
    plt.imshow(np.array(frame2[ymin:ymax, :]))
    plt.show(block=True)