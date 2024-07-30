import cv2
import supervision as sv
import os
from ultralytics import YOLO

# Path to the custom-trained model for helmet detection
model = YOLO("D:/Momentive_helmet-detection-yolov8/Momentive-helmet-detection-yolov8/models/hemletYoloV8_100epochs.pt")

# Open the video file
video_path = "g.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video was successfully opened
if not cap.isOpened():
    print(f"Error: Unable to open video file {video_path}")
    exit()

# Get video width and height
frame_wid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_hyt = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Loop through the frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Use the pre-trained YOLO model to detect helmets
    results = model(frame)[0]
    detections = results.boxes

    # Draw bounding boxes for detections
    for box in detections:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        confidence = box.conf.item()
        class_id = box.cls.item()

        # Draw only if class_id corresponds to "helmet" and confidence is above a threshold (e.g., 0.5)
        if model.names[class_id] == "helmet" and confidence > 0.5:
            # Draw a green rectangle around the detected helmet
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the frame with annotations
    cv2.imshow("Helmet Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
