import config
import numpy as np

def is_pixelated_difference_directional(image):
  """
  Checks if an image is pixelated by detecting repetitive patterns in a direction.

  Args:
    image: A NumPy array representing the image in BGR format.
    direction: Direction to calculate difference ("horizontal" or "vertical").

  Returns:
    True if the image is likely pixelated, False otherwise.
  """
  difference = np.abs(image[:, 1:] - image[:, :-1])

  # Calculate average difference
  avg_diff = np.mean(difference)

  return score_function(avg_diff)

def score_function(value):
    min_threshold = config.PIXELATED_MIN_THRESHOLD
    max_threshold = config.PIXELATED_MAX_THRESHOLD
    # If the value is below the min_threshold, return True
    if value < min_threshold:
        return True
    # If the value is above the max_threshold, return False
    elif value > max_threshold:
        return False
    else:
        # Calculate the position of the value in the range [min_threshold, max_threshold]
        normalized_value = (value - min_threshold) / (max_threshold - min_threshold)
        
        # Return True if normalized_value is less than 0.5, False otherwise
        return normalized_value < 0.5