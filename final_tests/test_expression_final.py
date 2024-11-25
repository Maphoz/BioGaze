from landmarks import landmark
import cv2
import numpy as np
import os
import pywt
import config
from metrics import compute_metrics
from emotion_recognizer import emotion_detector

land = landmark.LandmarkRecognizer()

emotion_recognizer = emotion_detector.EmotionDetector()

selected_directory = "./ICAO_selected"
wrong_directory = "./sm"


# Initialize y_true and y_pred as lists
y_true = []
y_pred = []

# Process the selected_directory
for filename in os.listdir(selected_directory):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(selected_directory, filename)
        mouth_open = land.mouth_check(img_path)
        emo_compliant = emotion_recognizer.check_neutral_expression(img_path)

        if emo_compliant and mouth_open >= config.MOUTH_THRESHOLD:
            y_pred.append(1)
        else:
            y_pred.append(0)
        y_true.append(1)  # All images in selected_directory are considered true positives

# Process the wrong_directory
for filename in os.listdir(wrong_directory):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(wrong_directory, filename)
        mouth_open = land.mouth_check(img_path)
        emo_compliant = emotion_recognizer.check_neutral_expression(img_path)

        if emo_compliant and mouth_open >= config.MOUTH_THRESHOLD:
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