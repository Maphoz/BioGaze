import cv2
import numpy as np
import os
import pywt
import config
from landmarks import landmark
from tests.metrics import compute_metrics

selected_directory = "./ICAO_selected"
wrong_directory = "./mkup"

# Initialize the FaceParser model
land = landmark.LandmarkRecognizer()

# Initialize y_true and y_pred as lists
y_true = []
y_pred = []

# Process the selected_directory
for filename in os.listdir(selected_directory):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(selected_directory, filename)
        has_makeup = land.makeup_check(img_path)

        if has_makeup:
            y_pred.append(0)
        else:
            y_pred.append(1)
        y_true.append(1)  # All images in selected_directory are considered true positives

# Process the wrong_directory
for filename in os.listdir(wrong_directory):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(wrong_directory, filename)
        has_makeup = land.makeup_check(img_path)

        if has_makeup:
            y_pred.append(0)
        else:
            y_pred.append(1)
        y_true.append(0)  # All images in wrong_directory are considered true negatives

# Convert y_true and y_pred to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Compute the Equal Error Rate (EER)
eer, _ = compute_metrics(y_pred, y_true, 0.1)

print('EER: ', eer)
