import streamlit as st
import cv2
import numpy as np
import yaml
import math
import time
import pygame
import pandas as pd
from datetime import datetime, timedelta
import subprocess  # For running data_saving.py as a subprocess

# Load YOLOv8 model
model_path = r"/best.pt"
data_yaml_path = r"/data.yaml"

# Load YOLOv8 using ultralytics
from ultralytics import YOLO
model = YOLO(model_path)

# Load data.yaml file
with open(data_yaml_path, 'r') as file:
    data_yaml = yaml.load(file, Loader=yaml.FullLoader)

class_names = model.names if hasattr(model, 'names') else data_yaml['names']

# Initialize statistics
statistics = {"mask_on_count": 0, "no_mask_on_count": 0, "total_people": 0}

# Initialize loop count
count_people_loop = 0

# Initialize minute-wise count variables
minute_start_time = time.time()
minute_count = 0

# Initialize pygame
pygame.init()

# Set the path to your alarm sound file
alarm_sound_path = "/MV27TES-alarm.mp3"
alarm_sound = pygame.mixer.Sound(alarm_sound_path)

# Flag to keep track of 'no mask on' state
previous_no_mask_state = False

# Flag to check if the alarm is playing
alarm_playing = False

# Initialize pandas DataFrame for data storage
data_columns = ["Timestamp", "Mask On Count", "No Mask On Count", "Total People"]
data = pd.DataFrame(columns=data_columns)

# Initialize session state to store data
if 'session_data' not in st.session_state:
    st.session_state.session_data = {"data": pd.DataFrame(columns=data_columns)}

# Button to switch between pages
selected_page = st.radio("Select Page", ["System Information", "Webcam Stream", "Statistics"])

# Start data saving script as a subprocess
if selected_page == "Webcam Stream":
    st.session_state.data_saving_process = None
    if st.button("Start Webcam"):
        st.session_state.data_saving_process = subprocess.Popen(
            ["python", "data_saving.py"],  # Change the command based on your environment
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

# Page 1: System Information
if selected_page == "System Information":
    st.title("🔬 LABWATCH 😷")
    st.write(
        "Welcome to LabWatch, the Real-time Mask Monitoring System for Laboratories. "
        "This system utilizes YOLOv8n, a powerful Computer Vision model, to detect whether workers "
        "in the laboratory are wearing masks. Below are some key details about the system:"
    )
    # Key Information
    st.subheader("Key Information ℹ️:")
    st.write(
        "- LabWatch employs YOLOv8n, an advanced Computer Vision model, for real-time mask detection."
    )
    st.write(
        "- It provides instant feedback on whether individuals in the laboratory are wearing masks."
    )
    st.write(
        "- The bounding boxes around workers are color-coded: 🟢 Green for 'mask on' and 🔴 Red for 'no mask on'."
    )
    st.write("- You can start and stop the webcam to observe the mask detection in action.")

    # Instructions
    st.subheader("Instructions 📋:")
    st.write(
        "1. Click the 'Start Webcam' button to activate the webcam and initiate real-time mask detection."
    )
    st.write(
        "2. The webcam stream will display in real-time, featuring color-coded bounding boxes around individuals."
    )
    st.write(
        "3. Bounding boxes are color-coded: 🟢 Green for 'mask on' and 🔴 Red for 'no mask on'."
    )
    st.write(
        "4. Click the 'Stop Webcam' button to deactivate the webcam when the monitoring session is complete."
    )

    # Additional Information
    st.subheader("Additional Information 🌐:")
    st.write(
        "- LabWatch is designed to ensure compliance with safety measures in the laboratory setting."
    )
    st.write(
        "- For inquiries or assistance, please contact the Laboratory Management. 📞"
    )

# Page 2: Webcam Stream
if st.session_state.data_saving_process:
    st.title("🔬 LABWATCH WEBCAM 😷")

    # Access the webcam only when the button is pressed
    cap = cv2.VideoCapture(0)

    # Display placeholder for the webcam stream
    video_placeholder = st.empty()

    # Display pandas table below the webcam
    data_placeholder = st.empty()

    # Button to stop webcam
    stop_button = st.button("Stop Webcam")
    stop_button_pressed = stop_button

    while cap.isOpened() and not stop_button_pressed:
        # Increment the loop count
        count_people_loop += 1
        
        ret, frame = cap.read()

        if not ret:
            st.warning("Unable to capture the frame.")
            break

        # Perform inference using YOLOv8
        results = model(frame, stream=True, conf=0.5, verbose=False)

        # Flag to check 'no mask on' state in the current frame
        current_no_mask_state = False

        for r in results:
            boxes = r.boxes

            for box in boxes:
                cls = int(box.cls[0])

                # Get the label of the instance
                label = class_names[cls]

                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                # Set box color based on class
                if label == "mask on":
                    box_color = (0, 255, 0)  # Green
                else:
                    box_color = (0, 0, 255)  # Red

                    # Set the flag for 'no mask on' state
                    current_no_mask_state = True

                # Put box in the webcam frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)

                # Update statistics
                statistics["total_people"] += 1

                if label == "mask on":
                    statistics["mask_on_count"] += 1
                else:
                    statistics["no_mask_on_count"] += 1

                # Display label on the frame
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Check if the current frame has 'no mask on' and the previous one did not
        if current_no_mask_state and not previous_no_mask_state:
            # Play the alarm sound for 'no mask on'
            pygame.mixer.Channel(0).play(alarm_sound)
            alarm_playing = True

        # Check if the current frame does not have 'no mask on'
        if not current_no_mask_state:
            # Stop the alarm sound if it is playing
            if alarm_playing:
                pygame.mixer.Channel(0).stop()
                alarm_playing = False

        # Update the previous state
        previous_no_mask_state = current_no_mask_state

        # Convert the frame to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the webcam stream with bounding boxes
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        # Update the data DataFrame with the current minute's statistics
        current_time = time.time()
        elapsed_time = current_time - minute_start_time

        if elapsed_time >= 10:
            # Add a new row for the current minute
            minute_count += 1
            minute_start_time = current_time

            # Calculate the real number of instances by dividing the counts by the loop count
            mask_on_count_real = round(statistics["mask_on_count"] / count_people_loop)
            no_mask_on_count_real = round(statistics["no_mask_on_count"] / count_people_loop)
            total_people_real = round(statistics["total_people"] / count_people_loop)

            # Store data in session state
            st.session_state.session_data["data"] = st.session_state.session_data["data"].append({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Mask On Count": mask_on_count_real,
                "No Mask On Count": no_mask_on_count_real,
                "Total People": total_people_real,
            }, ignore_index=True)

            # Reset minute-wise statistics and loop count
            statistics = {"mask_on_count": 0, "no_mask_on_count": 0, "total_people": 0}
            count_people_loop = 0

        # Display the data DataFrame in descending order based on "Timestamp"
        sorted_data = st.session_state.session_data["data"].sort_values(by="Timestamp", ascending=False)
        # Remove the 'Loop Count' column from the displayed DataFrame
        sorted_data = sorted_data.drop(columns=['Loop Count'], errors='ignore')
        data_placeholder.table(sorted_data)

    # Stop the pygame mixer channel if the alarm is still playing
    if alarm_playing:
        pygame.mixer.Channel(0).stop()

    # Release the webcam
    cap.release()
    st.success("Webcam Stopped")

# Page 3: Statistics
if selected_page == "Statistics":
    st.title("🔬 LABWATCH STATISTICS 😷")

    # Display stored data
    st.subheader("Real-time Data:")
    st.table(st.session_state.session_data["data"])

    # Example statistics plot
    st.subheader("Statistics Plot:")
    # Add your plot code here based on the stored data
    # Example plot (modify as needed):
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    # Choose a different color palette (e.g., "Set2")
    sns.set_palette("Set2")

    # Specify custom colors for each line
    mask_on_color = "skyblue"
    no_mask_color = "lightcoral"

    # Plot with rotated x-axis labels
    plt.plot(st.session_state.session_data["data"]["Timestamp"],
             st.session_state.session_data["data"]["Mask On Count"], label="Mask On Count", color=mask_on_color)
    plt.plot(st.session_state.session_data["data"]["Timestamp"],
             st.session_state.session_data["data"]["No Mask On Count"], label="No Mask On Count", color=no_mask_color)

    plt.xlabel("Timestamp")
    plt.ylabel("Count")
    plt.legend()

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90, ha="right")

    st.pyplot(plt)
