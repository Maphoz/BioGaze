import cv2
import numpy as np
import os
import pywt
import config
from landmarks import landmark
from face_parser import parserModel
from metrics import compute_metrics

selected_directory = "./ICAO_selected"
wrong_directory = "./light"

# Initialize the FaceParser model
land = landmark.LandmarkRecognizer()
parser = parserModel.FaceParser()

# Initialize y_true and y_pred as lists
y_true = []
y_pred = []

# Process the selected_directory
for filename in os.listdir(selected_directory):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(selected_directory, filename)
        uniform_luminosity = parser.detect_face_illumination(img_path)
        uniform_light = land.light_analysis(img_path)

        if uniform_luminosity and uniform_light:
            y_pred.append(1)
        else:
            y_pred.append(0)
        y_true.append(1)  # All images in selected_directory are considered true positives

# Process the wrong_directory
for filename in os.listdir(wrong_directory):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(wrong_directory, filename)
        uniform_luminosity = parser.detect_face_illumination(img_path)
        uniform_light = land.light_analysis(img_path)

        if uniform_luminosity and uniform_light:
            y_pred.append(1)
        else:
            y_pred.append(0)
        y_true.append(0)  # All images in wrong_directory are considered true negatives

# Convert y_true and y_pred to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Compute the Equal Error Rate (EER)
eer, _ = compute_metrics(y_pred, y_true, 0.1)

print('EER: ', eer)



'''
std 50 bright 80:
0.32

std 45 bright 80:
0.255


'''