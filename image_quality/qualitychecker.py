import config
import cv2
import numpy as np
from .pixelation_helper import is_pixelated_difference_directional
from .posterization_helper import analyze_rgb_channels

class QualityChecker:
    def __init__(self):
        return

    def is_out_of_focus(self, image_path):
      image = cv2.imread(image_path)
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      val = np.max(cv2.convertScaleAbs(cv2.Laplacian(gray, 3)))

      if val < config.MINIMUM_FOCUS_THRESHOLD:
         return True
      else:
         return False

    def is_pixelated(self, image_path):
      image = cv2.imread(image_path)
      is_pixelated_diff = is_pixelated_difference_directional(image)

      return is_pixelated_diff
    
    def is_posterized(self, image_path):
      image = cv2.imread(image_path)
      num_gaps = analyze_rgb_channels(image, gap_threshold=config.GAP_HISTOGRAM_THRESHOLD)

      if num_gaps > config.MAX_GAPS_THRESHOLD:
          return True
      
      return False