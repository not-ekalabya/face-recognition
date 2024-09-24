# Face Detection and Recognition System

This project is a real-time face detection and recognition system using YOLO for face detection and dlib for face recognition. It captures video frames from a webcam, detects faces using YOLO, and recognizes them by comparing with a database of known faces using dlib.

NOTE: Import DILB compressed files from https://github.com/z-mahmud22/Dlib_Windows_Python3.x

## Features
- **Real-time Face Detection**: YOLO model for real-time face detection.
- **Face Recognition**: dlib for computing and recognizing face encodings.
- **Data Augmentation**: Enhances accuracy with various image augmentations.

## How It Works
1. **Capture Images**: Capture and store images of known individuals.
2. **Load Known Faces**: Compute face encodings from stored images.
3. **Real-time Detection and Recognition**: Detect and recognize faces in real-time.

## YOLO Model
- **Single Convolutional Network**: Predicts bounding boxes and class probabilities.
- **Grid Division**: Divides image into grid cells predicting bounding boxes and confidence scores.
- **Class Probabilities**: Predicts class probabilities for each grid cell.
- **Non-Max Suppression**: Reduces overlapping boxes.

## Dlib for Face Recognition
- **Face Detection**: Uses HOG + Linear SVM model.
- **Face Landmark Detection**: Predicts facial landmarks.
- **Face Encoding**: Computes 128-dimensional face encoding.
- **Face Recognition**: Compares encodings with known faces.

## Integration of YOLO and Dlib
1. Capture frame from webcam.
2. Detect faces using YOLO.
3. Compute face encodings with dlib.
4. Compare encodings with known faces.
5. Identify or label as "Unknown".
6. Draw bounding boxes and names.
7. Display frame and repeat.

## Image Augmentation
Uses `imgaug` library for:
- Horizontal flipping
- Random cropping
- Gaussian blur
- Linear contrast adjustment
- Additive Gaussian noise
- Brightness adjustment
- Affine transformations (scaling, translation, rotation, shearing)

## Summary
- **YOLO**: Real-time face detection.
- **Dlib**: Face encoding and recognition.
- **Image Augmentation**: Improves robustness and performance.

This combination ensures efficient and accurate real-time face recognition.