import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import face_recognition
import os
import numpy as np

# Download YOLO model
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

# Load YOLO model
model = YOLO(model_path)

# Function to load known faces
def load_known_faces(data_folder):
    known_face_encodings = {}
    for person_folder in os.listdir(data_folder):
        person_path = os.path.join(data_folder, person_folder)
        if os.path.isdir(person_path):
            known_face_encodings[person_folder] = []
            for image_file in os.listdir(person_path):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(person_path, image_file)
                    face_image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(face_image)
                    if face_encodings:
                        known_face_encodings[person_folder].append(face_encodings[0])
    return known_face_encodings

# Load known faces
data_folder = 'data'
known_face_encodings = load_known_faces(data_folder)

# Function to recognize faces
def recognize_face(face_encoding, known_face_encodings, threshold=0.6):
    for name, encodings in known_face_encodings.items():
        matches = face_recognition.compare_faces(encodings, face_encoding, tolerance=threshold)
        if any(matches):
            return name
    return "Unknown"

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    
    if not ret:
        break

    # Convert frame to RGB (face_recognition uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces using YOLO
    results = model(rgb_frame)[0]

    # Get face locations
    face_locations = []
    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])
        face_locations.append((y1, x2, y2, x1))  # Convert to face_recognition format

    # Get face encodings
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Recognize faces
    face_names = []
    for face_encoding in face_encodings:
        name = recognize_face(face_encoding, known_face_encodings)
        face_names.append(name)

    # Draw bounding boxes and names on the frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if name == "Intruder":
            color = (0, 0, 255)  # Red for unknown faces
        else:
            color = (0, 255, 0)  # Green for known faces
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # Display the frame
    cv2.imshow('Face Detection and Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()