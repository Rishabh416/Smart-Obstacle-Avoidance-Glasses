from transformers import pipeline
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import time
from gtts import gTTS
from playsound import playsound
import os
import math

cap1 = cv2.VideoCapture(1) # left
cap2 = cv2.VideoCapture(0) # right

cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

orb = cv2.SIFT_create()

while True:
    ret, frame1 = cap1.read()
    ret, frame2 = cap2.read()

    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    frame1 = cv2.GaussianBlur(frame1,(5,5),0)
    frame2 = cv2.GaussianBlur(frame2,(5,5),0)

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
    x_end = min(frame1.shape[1], x + halflength)
    y_end = min(frame1.shape[0], y + halflength)
    gridTemplate = frame1[y_start:y_end, x_start:x_end]

    ymax = y + (2*halflength)
    ymin = y - (2*halflength)

    if ymax > 1920:
        ymax = 1920
    if ymin < 0:
        ymin = 0
    # -----------------------------------------
    keypoints1, descriptors1 = orb.detectAndCompute(gridTemplate, None)
    keypoints2, descriptors2 = orb.detectAndCompute(frame2[ymin:ymax, :], None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    matched_points1 = []
    matched_points2 = []
    points_distance = []

    for match in matches[:5]:  # Limit to the top 20 matches
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx

        # Get the coordinates of the keypoints in both images
        (x1, y1) = keypoints1[img1_idx].pt
        (x2, y2) = keypoints2[img2_idx].pt

        matched_points1.append((x1, y1))
        matched_points2.append((x2, y2))

    # Print the matched points
    print("Matched points in Image 1:")
    for point in matched_points1:
        pointX, pointY = point
        distance = math.sqrt(((x-pointX)**2)+((y-pointY)**2))
        points_distance.append(abs(distance))

    closestPoint = points_distance.index(min(points_distance))
    print(points_distance)
    print(f"closest point is {matched_points1[closestPoint]} and matching point is {matched_points2[closestPoint]}")

    matched_image = cv2.drawMatches(gridTemplate, keypoints1, frame2[ymin:ymax, :], keypoints2, matches[:5], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    f = plt.figure()
    f.add_subplot(1,1, 1)
    plt.imshow(np.array(matched_image))
    plt.show(block=True)
    # n1, m1 = matched_points1[closestPoint]
    # n2, m2 = matched_points2[closestPoint]
    # -----------------------------------------

    # result = cv2.matchTemplate(frame2[ymin:ymax, :], gridTemplate, cv2.TM_CCOEFF) # TM_SQDIFF TM_CCOEFF
    # min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(result)
    # print("image2",max_val2, max_loc2)
    # c, h, w = gridTemplate.shape[::-1]
    # x_centerLoc = max_loc2[0] + w // 2

    pixelDistance = (0)
    objectDistance = round((0.0004*(pixelDistance**2))+(0.3767*(pixelDistance))+110.8) # todo: update equation for object distance upto 2 m at 10cm intervals
    print(objectDistance)

    h, w, c = frame1.shape

    vertical = "middle"
    horizontal = "middle"

    match (x // (w/3)):
        case 0.0:
            horizontal = "left"
        case 1.0:
            horizontal = "middle"
        case 2.0:
            horizontal = "right"

    match (y // (h/3)):
        case 0.0:
            horizontal = "top"
        case 1.0:
            horizontal = "middle"
        case 2.0:
            horizontal = "bottom"

    text = f'closest object at {objectDistance} centimeters, direction {vertical},{horizontal}' 
    print(text)

    # text to speech setup
    # tts = gTTS(text=text, lang='en')
    # tts.save("audio.mp3")
    # playsound("audio.mp3")
    # os.remove("audio.mp3") 

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break