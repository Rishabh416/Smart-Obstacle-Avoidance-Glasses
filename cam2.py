import cv2
import tkinter as tk
from tkinter import Scale, HORIZONTAL
from PIL import Image, ImageTk
import numpy as np
from multiprocessing import shared_memory

# Define the shared memory name, shape, and dtype for Camera 2
shared_mem_name = 'camera2_shared_mem'
frame_shape = (480, 640, 3)  # Assuming 640x480 resolution with 3 channels (RGB)
dtype = np.uint8  # 8-bit unsigned integer (for image data)

# Create a shared memory block for Camera 2 feed (only need to do this once)
shm = shared_memory.SharedMemory(create=True, size=int(np.prod(frame_shape)) * np.dtype(dtype).itemsize, name=shared_mem_name)

# Create a NumPy array backed by shared memory
shared_array = np.ndarray(frame_shape, dtype=dtype, buffer=shm.buf)

# Initialize brightness and contrast factors
brightness_factor = 1.0
contrast_factor = 1.0

# Function to update webcam feed in the Tkinter window
def update_frame():
    ret, frame = cap.read()  # Capture frame from the webcam
    if not ret:
        return

    # Resize frame to match the shared memory shape if necessary (e.g., 640x480)
    frame_resized = cv2.resize(frame, (frame_shape[1], frame_shape[0]))

    # Apply brightness and contrast adjustments
    adjusted_frame = cv2.convertScaleAbs(frame_resized, alpha=contrast_factor, beta=(brightness_factor - 1) * 255)

    # Write the adjusted frame to shared memory
    shared_array[:] = adjusted_frame[:]

    # Convert the frame to a Tkinter-compatible image
    img = Image.fromarray(adjusted_frame)
    imgtk = ImageTk.PhotoImage(image=img)

    # Update the label with the new frame
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Call the update_frame function after 10 milliseconds
    video_label.after(10, update_frame)

# Function to adjust the focus property of the camera
def adjust_focus(value):
    cap.set(cv2.CAP_PROP_FOCUS, int(value))

# Function to adjust the synthetic brightness based on the slider value
def adjust_brightness(value):
    global brightness_factor
    brightness_factor = float(value) / 100  # Convert slider value to a factor between 0 and 2

# Function to adjust the synthetic contrast based on the slider value
def adjust_contrast(value):
    global contrast_factor
    contrast_factor = float(value) / 100  # Convert slider value to a factor between 0 and 2

# Function to release the camera, close the shared memory, and close the window when quitting
def quit_program():
    cap.release()  # Release the webcam
    shm.close()    # Close the shared memory block
    shm.unlink()   # Unlink shared memory (delete it)
    root.quit()    # Close the Tkinter window

# Set up the Tkinter window
root = tk.Tk()
root.title("Camera 2 with Focus, Brightness, and Contrast Control")

# Set up the webcam capture using OpenCV (use index 1 for Camera 2)
cap = cv2.VideoCapture(1)

# Check if camera supports manual focus
if not cap.get(cv2.CAP_PROP_AUTOFOCUS):
    print("Manual focus supported. Autofocus is turned off.")
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus

# Create a label to display the video feed
video_label = tk.Label(root)
video_label.pack()

# Create a slider to adjust the focus
focus_slider = Scale(root, from_=0, to=255, orient=HORIZONTAL, label="Focus")
focus_slider.set(int(cap.get(cv2.CAP_PROP_FOCUS)))  # Set the slider to the current focus level
focus_slider.pack()

# Bind the slider to the focus adjustment function
focus_slider.config(command=adjust_focus)

# Create a slider to adjust the synthetic brightness
brightness_slider = Scale(root, from_=0, to=200, orient=HORIZONTAL, label="Brightness")
brightness_slider.set(100)  # Set the default brightness level (100%)
brightness_slider.pack()

# Bind the slider to the brightness adjustment function
brightness_slider.config(command=adjust_brightness)

# Create a slider to adjust the synthetic contrast
contrast_slider = Scale(root, from_=0, to=200, orient=HORIZONTAL, label="Contrast")
contrast_slider.set(100)  # Set the default contrast level (100%)
contrast_slider.pack()

# Bind the slider to the contrast adjustment function
contrast_slider.config(command=adjust_contrast)

# Start the video capture loop
update_frame()

# Bind the quit function to the window close button
root.protocol("WM_DELETE_WINDOW", quit_program)

# Start the Tkinter main loop
root.mainloop()
