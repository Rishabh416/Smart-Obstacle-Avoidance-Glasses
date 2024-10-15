import cv2
import tkinter as tk
from tkinter import Scale, HORIZONTAL
from PIL import Image, ImageTk

# Function to update webcam feed in the Tkinter window
def update_frame():
    ret, frame = cap.read()  # Capture frame from the webcam
    if not ret:
        return

    # Convert the frame to RGB (OpenCV uses BGR by default)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to a Tkinter-compatible image
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)

    # Update the label with the new frame
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Call the update_frame function after 10 milliseconds
    video_label.after(10, update_frame)

# Function to adjust the focus property of the camera
def adjust_focus(value):
    cap.set(cv2.CAP_PROP_FOCUS, int(value))

# Function to release the camera and close the window when quitting
def quit_program():
    cap.release()  # Release the webcam
    root.quit()    # Close the Tkinter window

# Set up the Tkinter window
root = tk.Tk()
root.title("Webcam with Focus Control")

# Set up the webcam capture using OpenCV
cap = cv2.VideoCapture(0)

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

# Start the video capture loop
update_frame()

# Bind the quit function to the window close button
root.protocol("WM_DELETE_WINDOW", quit_program)

# Start the Tkinter main loop
root.mainloop()
