import cv2
import supervision as sv
import os
from datetime import datetime
import sys

from utils.helperFunctions import *
from ultralytics import YOLO

# Path to the custom trained model
model = YOLO("D:/Momentive_helmet-detection-yolov8/Momentive-helmet-detection-yolov8/models/hemletYoloV8_100epochs.pt")

# Video source and output settings
video_path = "f.mp4"
output_video_path = "output_annotated_video.mp4"

# Frame width and height
frame_wid = 640
frame_hyt = 480

def processVideo(video_path, output_video_path):
    """
    Process a video, detect helmets using a pre-trained YOLOv8 model,
    and store the annotated video.

    Args:
        - video_path: Path to the input video.
        - output_video_path: Path to save the annotated video.
    """
    # Initialize video capture and get video properties
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Set up video writer for saving the annotated video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    box_annotator = sv.BoxAnnotator(thickness=2)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame (optional)
        image = cv2.resize(frame, (frame_wid, frame_hyt))

        # Detect helmets in the frame
        results = model(image)[0]

        # Extract bounding boxes, confidences, and class IDs
        boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        # Create labels and annotate the image
        labels = [f"{model.model.names[class_id]} {confidence:.2f}" for class_id, confidence in zip(class_ids, confidences)]

        # Annotate the frame with bounding boxes
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if "helmet" in label else (0, 0, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write the annotated frame to the output video
        out.write(image)

        # Optionally display the frame
        # cv2.imshow("Helmet Detection", image)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release resources
    cap.release()
    out.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    """
    Main function for processing the video to detect helmets and save the annotated video.
    """
    try:
        # Process the video and save the annotated output
        processVideo(video_path, output_video_path)
        print(f"Annotated video saved to '{output_video_path}'")
    except Exception as error:
        print(f"[!] An error occurred: {error}")
