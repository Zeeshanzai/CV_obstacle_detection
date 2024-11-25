import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import pyttsx3  # Library for text-to-speech

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speech rate
engine.setProperty('volume', 0.9)  # Adjust volume

# Download the MiDaS model
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.eval()

# Input transformation pipeline
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

# Initialize webcam and reduce resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Set up Matplotlib figure
plt.ion()  # Enable interactive mode
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].set_title("Webcam Feed")
axes[1].set_title("Depth Map")

frame_skip = 2
frame_count = 0

# Store last spoken feedback to avoid repeating
last_feedback = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Resize frame for MiDaS
    frame = cv2.resize(frame, (320, 240))
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img)

    # Make a prediction
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(120, 160),  # Half-resolution depth map
            mode='bicubic',
            align_corners=False
        ).squeeze()
        output = prediction.cpu().numpy()

    # Normalize depth map for better visualization
    depth_map = cv2.normalize(output, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Analyze depth map to provide text feedback
    near_threshold = 60  # Define near object threshold
    medium_threshold = 120  # Define medium-distance threshold

    avg_depth = np.mean(depth_map)
    if avg_depth < near_threshold:
        text_feedback = "Objects are near!"
    elif avg_depth < medium_threshold:
        text_feedback = "Objects are at a medium distance."
    else:
        text_feedback = "Objects are far away."

    # Speak feedback only if it changes
    if text_feedback != last_feedback:
        engine.say(text_feedback)
        engine.runAndWait()
        last_feedback = text_feedback

    print(f"Average Depth: {avg_depth:.2f} | {text_feedback}")  # Print feedback in terminal

    # Update plots
    axes[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0].text(10, 20, text_feedback, color="red", fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    axes[1].imshow(depth_map, cmap='inferno')
    plt.pause(0.01)  # Pause for interactive display

# Release resources
cap.release()
plt.close()
