from transformers import pipeline
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
# import time
# from gtts import gTTS
# from playsound import playsound
# import os
# import math
from multiprocessing import shared_memory

# Define the shared memory block details
frame_shape = (480, 640, 3)  # Assuming 640x480 resolution with 3 channels (RGB)
dtype = np.uint8  # Image data is 8-bit unsigned integers

# Connect to the shared memory for Camera 1 and Camera 2
shm1 = shared_memory.SharedMemory(name='camera1_shared_mem')
shm2 = shared_memory.SharedMemory(name='camera2_shared_mem')

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

orb = cv2.SIFT_create()


rainbow_overlay = np.zeros((480, 640, 3), dtype=np.uint8)

# Define the colors for the rainbow (Red to Purple in BGR format)
rainbow_colors = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 255, 0),  # Cyan
    (255, 0, 255)   # Magenta
]

# Calculate the width of each color band
band_width = 640 // (len(rainbow_colors) - 1)

for i in range(len(rainbow_colors) - 1):
    for x in range(i * band_width, (i + 1) * band_width):
        # Interpolate between two colors
        alpha = (x - i * band_width) / band_width
        color = (1 - alpha) * np.array(rainbow_colors[i]) + alpha * np.array(rainbow_colors[i + 1])
        rainbow_overlay[:, x, :] = color

while True:

    # Create NumPy arrays backed by the shared memory for each camera
    frame2 = np.ndarray(frame_shape, dtype=dtype, buffer=shm1.buf)
    frame1 = np.ndarray(frame_shape, dtype=dtype, buffer=shm2.buf)

    # Now frame1 and frame2 represent the images from Camera 1 and Camera 2
    # Example usage
    image1 = np.copy(frame1)  # Image from Camera 1
    image2 = np.copy(frame2)  # Image from Camera 2

    frame1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    frame2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    result = pipe(image)
    depthimage = result["depth"]

    depthimagearray = np.array(depthimage)
    min1, max1, micloc1, maxloc1 = cv2.minMaxLoc(depthimagearray)
    print("image1",max1, maxloc1)
    x, y = maxloc1
    
    opacity = 0.6
    frame1 = cv2.addWeighted(rainbow_overlay, opacity, frame1, 1 - opacity, 0)
    frame2 = cv2.addWeighted(rainbow_overlay, opacity, frame2, 1 - opacity, 0)

    halflength = 100
    x_start = max(0, x - halflength)
    y_start = max(0, y - halflength)
    x_end = min(frame1.shape[1], x + halflength)
    y_end = min(frame1.shape[0], y + halflength)
    gridTemplate = frame1[y_start:y_end, x_start:x_end]

    ymax = y + (2*halflength)
    ymin = y - (2*halflength)

    if ymax > 640:
        ymax = 640
    if ymin < 0:
        ymin = 0
    
    try:
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

        matched_image = cv2.drawMatches(gridTemplate, keypoints1, frame2[ymin:ymax, :], keypoints2, matches[:1], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        f = plt.figure()
        f.add_subplot(1,1, 1)
        plt.imshow(np.array(matched_image))
        plt.show(block=True)

        print(f"closest point is {matched_points1[0]} and matching point is {matched_points2[0]}")
        pointX1, pointY1 = matched_points1[0]
        pointX2, pointY2 = matched_points2[0]
        print("pixel distance: ", (pointX1 + x_start)-(pointX2))

        pixelDistance = ((pointX1 + x_start)-(pointX2))
        objectDistance = round((178.234)*(0.982881**(pixelDistance))+100.1429) # todo: update equation for object distance upto 2 m at 10cm intervals
        print(objectDistance)

        h, w, c = frame1.shape

        vertical = "middle"
        horizontal = "middle"

        match (pointX2 // (w/3)):
            case 0.0:
                horizontal = "left"
            case 1.0:
                horizontal = "middle"
            case 2.0:
                horizontal = "right"

        match ((pointY2 + y_start) // (h/3)):
            case 0.0:
                horizontal = "top"
            case 1.0:
                horizontal = "middle"
            case 2.0:
                horizontal = "bottom"

        text = f'closest object at {objectDistance} centimeters, direction {vertical},{horizontal}' 
        # print(text)

        # text to speech setup
        # tts = gTTS(text=text, lang='en')
        # tts.save("audio.mp3")
        # playsound("audio.mp3")
        # os.remove("audio.mp3") 

    except:
        print("error")

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break