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
import math

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
focalLength = 16 # f
camPixelSize = 0.001875 # d 3.6/1920
camDistance = 60 # T

# relative depth estimation model
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

orb = cv2.ORB_create()

while True:
    # capture image from both cameras
    ret, frame1 = cap1.read()
    ret, frame2 = cap2.read()
    
    # image pre processing, smoothing camera noise
    frame1blur = cv2.GaussianBlur(frame1,(3,3),0)
    frame2blur = cv2.GaussianBlur(frame2,(3,3),0)
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

    # find matching template in image from camera 2
    keypoints1, descriptors1 = orb.detectAndCompute(frame1blur, None)
    keypoints2, descriptors2 = orb.detectAndCompute(frame2blur, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    matched_points1 = []
    matched_points2 = []
    points_distance = []

    for match in matches[:500]:  # Limit to the top 20 matches
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

    matched_image = cv2.drawMatches(frame1blur, keypoints1, frame2blur, keypoints2, matches[:500], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    f = plt.figure()
    f.add_subplot(1,1, 1)
    plt.imshow(np.array(matched_image))
    plt.show(block=True)

    n1, m1 = matched_points1[closestPoint]
    n2, m2 = matched_points2[closestPoint]

    # calculate the distance of nearest object
    pixelDistance = (n1 - n2) # n1-n2
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