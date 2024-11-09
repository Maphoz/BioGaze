import cv2
from rmn import RMN  # Assuming this is the model you're using.
import config
import os

class EmotionDetector:
    def __init__(self):
        # Initialize the RMN model (or any other emotion detection model).
        self.model = RMN()

    def check_neutral_expression(self, img_path):
        """
        Detects emotion from the given image and checks if the emotion is neutral and the second most
        likely emotion has a low probability.

        Parameters:
            img_path (str): Path to the image.

        Returns:
            bool: True if the emotion is neutral, False otherwise.
        """
        # Read the image.
        image = cv2.imread(img_path)
    
        # Use the model to detect emotion.
        results = self.model.detect_emotion_for_single_frame(image)

        # Extract most probable emotion label and probability
        emo_label = results[0]['emo_label']
        emo_most_proba = round(results[0]['emo_proba'], 3)

        # Extract the second most probable emotion's probability
        proba_list = results[0]['proba_list']
        sorted_proba = sorted([list(item.values())[0] for item in proba_list], reverse=True)
        second_most_proba = round(sorted_proba[1], 3)

        # Calculate the ratio between the second most probable and the most probable emotion
        emo_rat = second_most_proba / emo_most_proba

        # Determine the suffix based on conditions
        if emo_label == "neutral" and emo_rat < config.MAX_RATIO_EMOTION:
            return True
        else:
            return False
        
    def draw_emotion(self, img_path):
        image = cv2.imread(img_path)
        results = self.model.detect_emotion_for_single_frame(image)
        image = self.model.draw(image, results)
        output_path = img_path.replace('.jpg', '_emo.jpg').replace('.png', '_emo.png').replace('.JPG', '_emo.JPG')
        cv2.imwrite(output_path, image)
    