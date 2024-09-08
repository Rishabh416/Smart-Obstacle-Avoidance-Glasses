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
cap2 = cv2.VideoCapture(2) # right

cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

orb = cv2.ORB_create()

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
    
    keypoints1, descriptors1 = orb.detectAndCompute(gridTemplate, None)
    keypoints2, descriptors2 = orb.detectAndCompute(frame2[ymin:ymax, :], None)

    index_params = dict(algorithm=6,  # FLANN_INDEX_LSH (used for ORB, BRIEF, BRISK)
                    table_number=6,  # 12
                    key_size=12,     # 20
                    multi_probe_level=1)  # 2
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=1)

    distance_threshold = 0.75  # Threshold for cluster distance (you can adjust this)
    filtered_matches = []

    # Cluster-based filtering
    for m in matches:
        # Extract the distances of the k nearest neighbors
        distances = [match.distance for match in m]
        
        # Calculate the mean distance
        mean_distance = np.mean(distances)
        
        # Calculate the standard deviation of distances
        std_distance = np.std(distances)
        
        # Filter based on the standard deviation being below a threshold
        if std_distance / mean_distance < distance_threshold:
            # Consider the match as valid and append the best match (first in the list)
            filtered_matches.append(m[0])

    # Display or use filtered_matches
    print(f"Total matches before filtering: {len(matches)}")
    print(f"Total matches after filtering: {len(filtered_matches)}")

    # Optionally, you can draw the matches on images
    img_matches = cv2.drawMatches(gridTemplate, keypoints1, frame2[ymin:ymax, :], keypoints2,
        filtered_matches[:1], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)    

    f = plt.figure()
    f.add_subplot(1,1, 1)
    plt.imshow(np.array(img_matches))
    plt.show(block=True)

    # print(f"closest point is {matched_points1[0]} and matching point is {matched_points2[0]}")
    # pointX1, pointY1 = matched_points1[0]
    # pointX2, pointY2 = matched_points2[0]
    # print((pointX1 + x_start)-(pointX2))

    # pixelDistance = ((pointX1 + x_start)-(pointX2))
    # objectDistance = round((0.00869691*(pixelDistance**2))+(-5.67746*(pixelDistance))+938.723) # todo: update equation for object distance upto 2 m at 10cm intervals
    # print(objectDistance)

    # h, w, c = frame1.shape

    # vertical = "middle"
    # horizontal = "middle"

    # match (pointX2 // (w/3)):
    #     case 0.0:
    #         horizontal = "left"
    #     case 1.0:
    #         horizontal = "middle"
    #     case 2.0:
    #         horizontal = "right"

    # match ((pointY2 + y_start) // (h/3)):
    #     case 0.0:
    #         horizontal = "top"
    #     case 1.0:
    #         horizontal = "middle"
    #     case 2.0:
    #         horizontal = "bottom"

    # text = f'closest object at {objectDistance} centimeters, direction {vertical},{horizontal}' 
    # print(text)

    # # text to speech setup
    # tts = gTTS(text=text, lang='en')
    # tts.save("audio.mp3")
    # playsound("audio.mp3")
    # os.remove("audio.mp3") 
    

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break