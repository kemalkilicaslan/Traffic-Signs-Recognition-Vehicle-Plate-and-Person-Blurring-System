# Traffic Signs Recognition, Vehicle Plate and Person Blurring System

# Import the necessary libraries
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import cv2
import config

# Paths (file names loaded from config.py, resolved relative to this script's folder)
BASE_DIR = Path(__file__).resolve().parent

# Import the necessary models
vehicle_plate_detection_model = YOLO(str(BASE_DIR / config.VEHICLE_PLATE_MODEL_PATH))  # Model for vehicle plate detection
traffic_signs_detection_model = YOLO(str(BASE_DIR / config.TRAFFIC_SIGNS_MODEL_PATH))  # Model for traffic signs detection
YOLO_model = YOLO(str(BASE_DIR / config.YOLO_MODEL_PATH))  # Load YOLOv11 model for person, car, motorcycle, bus, truck detection

# Open video file
video = str(BASE_DIR / config.INPUT_VIDEO)
cap = cv2.VideoCapture(video)

# Get video resolution and FPS value
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create a VideoWriter object
output = str(BASE_DIR / config.OUTPUT_VIDEO)

# Starts VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
out = cv2.VideoWriter(output, fourcc, fps, (frame_width, frame_height))

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Traffic sign detection
    traffic_signs_results = traffic_signs_detection_model(frame, conf=config.TRAFFIC_SIGN_CONFIDENCE)
    for result in traffic_signs_results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs
        confidences = result.boxes.conf.cpu().numpy()  # Accuracy rates

        for box, class_id, confidence in zip(boxes, class_ids, confidences):
            x1, y1, x2, y2 = map(int, box)
            # Get class name and accuracy rate
            class_name = traffic_signs_detection_model.names[int(class_id)]
            label = f"{class_name} {confidence:.2f}"

            # Calculate text size
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, config.LABEL_FONT_SCALE, config.LABEL_FONT_THICKNESS)

            # Center the text at the top of the bounding box
            text_x = x1 + (x2 - x1 - text_width) // 2  # Center text horizontally
            text_y = y1 - 10  # Place text on top of the bounding box

            # Draw rectangle for text background
            cv2.rectangle(frame, (text_x, text_y - text_height), (text_x + text_width, text_y), config.LABEL_BG_COLOR, -1)  # Background color

            # Write the text
            cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, config.LABEL_FONT_SCALE, config.LABEL_TEXT_COLOR, config.LABEL_FONT_THICKNESS)  # Text color black

            # Draw a bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), config.BOX_COLOR, config.BOX_THICKNESS)

    # Person detection with YOLOv11 (confidence threshold from config.py)
    yolov11_results = YOLO_model(frame, conf=config.DETECTION_CONFIDENCE)
    for result in yolov11_results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

        for box, class_id in zip(boxes, class_ids):
            class_name = YOLO_model.names[int(class_id)]
            x1, y1, x2, y2 = map(int, box)

            # Blur the entire person if class is 'person'
            if class_name == config.PERSON_CLASS:
                # Extract the person region
                person_region = frame[y1:y2, x1:x2]

                # Apply blur to the entire person
                blurred_person = cv2.GaussianBlur(person_region, config.PERSON_BLUR_KERNEL, config.PERSON_BLUR_SIGMA)  # Adjust blur strength as needed
                frame[y1:y2, x1:x2] = blurred_person

            # Detect and blur vehicle plates inside vehicle bounding boxes
            if class_name in config.VEHICLE_CLASSES:
                # Crop the vehicle region
                vehicle_region = frame[y1:y2, x1:x2]

                # Detect vehicle plates within the vehicle region (confidence threshold from config.py)
                plate_results_in_vehicle = vehicle_plate_detection_model(vehicle_region, conf=config.PLATE_CONFIDENCE)
                for plate_result in plate_results_in_vehicle:
                    plate_boxes = plate_result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
                    for plate_box in plate_boxes:
                        px1, py1, px2, py2 = map(int, plate_box)
                        # Adjust coordinates to the original frame
                        px1 += x1
                        py1 += y1
                        px2 += x1
                        py2 += y1

                        # Blurring the vehicle plate area
                        plate_region = frame[py1:py2, px1:px2]
                        blurred_plate = cv2.GaussianBlur(plate_region, config.PLATE_BLUR_KERNEL, config.PLATE_BLUR_SIGMA)  # Adjust blur strength as needed
                        frame[py1:py2, px1:px2] = blurred_plate

    if config.SHOW_WINDOW:
        cv2.imshow('Traffic Signs Recognition, Vehicle Plate and Person Blurring System', frame)  # Shows the image
    out.write(frame)  # Adds the frame to VideoWriter

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Checks whether the 'q' key has been pressed to exit
        break

cap.release()
out.release()  # Close VideoWriter
cv2.destroyAllWindows()  # Closes the OpenCV window
