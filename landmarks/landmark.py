import os
import config
import cv2
import dlib
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def process_boxes(box):
    xmin = box.left()
    ymin = box.top()
    xmax = box.right()
    ymax = box.bottom()
    return [int(xmin), int(ymin), int(xmax), int(ymax)]

def point_distance(a, b):
  pointA = (a.x, a.y)
  pointB = (b.x, b.y) 
  distance = math.sqrt((pointA[0] - pointB[0]) ** 2 + (pointA[1] - pointB[1]) ** 2)

  return distance

def math_distance(aX, aY, bX, bY):
  return math.sqrt((aX - bX) ** 2 + (aY - bY) ** 2)

def calculate_mean_intensity(image, top_left, bottom_right):
    x1, y1 = top_left
    x2, y2 = bottom_right
    region = image[y1:y2, x1:x2]
    mean_intensity = cv2.mean(region)[:3]  # Return mean values for the BGR channels
    return mean_intensity

def calculate_EVZ(shape, eye_start, eye_end, IED_margin, SUN_margin):
    '''
    calculate the EVZ area around the yes in accordance to the ICAO suggestions.

    It returns the EVZ, of the dimension suggested by the ICAO
    It also returns another area, and the dimension here is decidable by the user (with SUN_margin)
    '''
    # Extract eye points from the shape object
    eye_points = [(shape.part(i).x, shape.part(i).y) for i in range(eye_start, eye_end + 1)]
    
    # Get the x and y coordinates separately
    x_coords = [point[0] for point in eye_points]
    y_coords = [point[1] for point in eye_points]

    # Calculate the minimum and maximum x and y coordinates
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Create dimensions with IED margin
    x_min_IED = x_min - IED_margin
    x_max_IED = x_max + IED_margin
    y_min_IED = y_min - IED_margin
    y_max_IED = y_max + IED_margin

    IED_area = (x_min_IED, y_min_IED, x_max_IED, y_max_IED)

    # Create dimensions with SUN margin
    x_min_SUN = x_min - SUN_margin
    x_max_SUN = x_max + SUN_margin
    y_min_SUN = y_min - SUN_margin
    y_max_SUN = y_max + SUN_margin

    SUN_area = (x_min_SUN, y_min_SUN, x_max_SUN, y_max_SUN)

    # Return both sets of coordinates
    return IED_area, SUN_area

def check_makeup(image, area1, area2):

    def has_makeup(image):
        #print(num)
        # Convert the cropped image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Extract the H(hue) channel
        h_channel = hsv_image[:, :, 0]
        
        # Calculate the histogram of the V channel
        hist = cv2.calcHist([h_channel], [0], None, [180], [0, 180])
        
        # Normalize the histogram
        hist = hist / hist.sum()

        high_hue = hist[50:180].sum()

        return high_hue
    
   # Extract and check the first area
    x_min, y_min, x_max, y_max = area1
    cropped_image1 = image[y_min:y_max, x_min:x_max]
    l_makeup = has_makeup(cropped_image1)

    # Extract and check the second area
    x_min, y_min, x_max, y_max = area2
    cropped_image2 = image[y_min:y_max, x_min:x_max]
    r_makeup = has_makeup(cropped_image2)

    return l_makeup > config.MAKEUP_HIGH_HUE_THRESHOLD and r_makeup > config.MAKEUP_HIGH_HUE_THRESHOLD

def check_red_eye(image, EVZ):
    """
    Function to check for red-eye effect in an eye region (EVZ).

    Args:
        image_path: Path to the image.
        EVZ: A tuple representing the eye region coordinates (x_min, y_min, x_max, y_max).

    Returns:
        red_pixel_count: Number of red pixels in the region.
        red_pixel_ratio: Ratio of red pixels in the region.
    """

    # Read the image
    x_min, y_min, x_max, y_max = EVZ
    evz_region = image[y_min:y_max, x_min:x_max]

    # Convert to HSV color space
    hsv_evz = cv2.cvtColor(evz_region, cv2.COLOR_BGR2HSV)

    # Blur to reduce noise
    blurred_hsv = cv2.GaussianBlur(hsv_evz, (3, 3), 0)

    # Red color detection in HSV
    lower_red_1 = np.array([0, 50, 70], dtype="uint8")  # Adjusted lower bound
    upper_red_1 = np.array([10, 255, 255], dtype="uint8")
    lower_red_2 = np.array([170, 50, 70], dtype="uint8")  # Adjusted lower bound
    upper_red_2 = np.array([180, 255, 255], dtype="uint8")
    red_mask_hsv = cv2.inRange(blurred_hsv, lower_red_1, upper_red_1) + cv2.inRange(blurred_hsv, lower_red_2, upper_red_2)

    # Morphological operations for noise reduction and hole filling
    kernel = np.ones((5, 5), np.uint8)
    red_mask_hsv = cv2.morphologyEx(red_mask_hsv, cv2.MORPH_OPEN, kernel, iterations=3)
    red_mask_hsv = cv2.morphologyEx(red_mask_hsv, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Calculate red pixel ratio
    red_pixel_count = np.sum(red_mask_hsv > 0)
    red_pixel_ratio = red_pixel_count / red_mask_hsv.size

    # Threshold adjustment based on empirical data
    threshold = 0.1  # Adjust threshold based on your dataset
    is_red_eye = red_pixel_ratio > threshold

    return is_red_eye

def get_mean_color(image, area):
    x_min, y_min, x_max, y_max = area
    cropped_image = image[y_min:y_max, x_min:x_max]
    mean_color = np.mean(cropped_image[:, :, 0])
    return mean_color

def hue_distance(hue1, hue2):
    # Calculate the difference in hue values, accounting for the circular nature of hue
    diff = np.abs(hue1 - hue2)
    return min(diff, 180 - diff)

def check_makeup_distance(image, skin_area1, skin_area2, eye_area1, eye_area2, threshold=config.MAKEUP_DISTANCE_THRESHOLD):
    # Convert image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    skin_area1 = get_mean_color(hsv_image, skin_area1)
    skin_area2 = get_mean_color(hsv_image, skin_area2)

    mean_skin_area = (skin_area1 + skin_area2) / 2

    # Get mean color of the eye regions
    eye_color1 = get_mean_color(hsv_image, eye_area1)
    eye_color2 = get_mean_color(hsv_image, eye_area2)

    # Calculate color distance
    distance1 = hue_distance(mean_skin_area, eye_color1)
    distance2 = hue_distance(mean_skin_area, eye_color2)

    # Determine if makeup is present based on the threshold
    makeup_detected = (distance1 > threshold) and (distance2 > threshold)
    return makeup_detected

class LandmarkRecognizer:
  def __init__(self):
    """
    Initializes the face detector with the specified model path.
    """
    predictor_path = './dlib_checkpoint/shape_predictor_68_face_landmarks.dat'
    self.detector = dlib.get_frontal_face_detector()
    self.predictor = dlib.shape_predictor(predictor_path)

  def makeup_check(self, image_path):
    '''
    Individual check to verify the presence of excessive makeup.

    Returns:
      a boolean that is true if one of the two makeup checks results positive.
    '''
    image = cv2.imread(image_path)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detected_boxes = self.detector(image_rgb)
    box = detected_boxes[0]

    shape = self.predictor(image_rgb, box)

    '''
    CALCULATION OF IED
    '''

    left_eye_start = 36
    left_eye_end = 39
    right_eye_start = 42
    right_eye_end = 45

    # Calculate middle points of each eye
    left_eye_middle_x = (shape.part(left_eye_start).x + shape.part(left_eye_end).x) / 2
    left_eye_middle_y = (shape.part(left_eye_start).y + shape.part(left_eye_end).y) / 2

    right_eye_middle_x = (shape.part(right_eye_start).x + shape.part(right_eye_end).x) / 2
    right_eye_middle_y = (shape.part(right_eye_start).y + shape.part(right_eye_end).y) / 2

    middle_x = (left_eye_middle_x + right_eye_middle_x) / 2
    middle_y = (left_eye_middle_y + right_eye_middle_y) / 2


    # Calculate inter-eye distance (IED) between middle points
    inter_eye_distance = math_distance(left_eye_middle_x, left_eye_middle_y, right_eye_middle_x, right_eye_middle_y)
        
    mouth_up = shape.part(62)
    mouth_down = shape.part(66)
        
    # Calculate the midpoint
    mouth_mid_x = (mouth_up.x + mouth_down.x) / 2
    mouth_mid_y = (mouth_up.y + mouth_down.y) / 2
    side_length = 0.3 * inter_eye_distance

    '''
    CALCULATION OF LUMINOSITY
    '''

    emd = math_distance(middle_x, middle_y, mouth_mid_x, mouth_mid_y)

    '''CHEEK SQUARES'''

    P_x = middle_x
    P_y = middle_y + (emd / 2)

    # Calculate the top-right corner for the right cheek square (0.5 * IED to the left from P)
    right_cheek_top_right_x = P_x - (0.5 * inter_eye_distance)
    right_cheek_top_right_y = P_y

    # Calculate the coordinates of the right cheek square
    top_left_r = (int(right_cheek_top_right_x - side_length), int(right_cheek_top_right_y))
    bottom_right_r = (int(right_cheek_top_right_x), int(right_cheek_top_right_y + side_length))

    # Calculate the top-left corner for the left cheek square (0.5 * IED to the right from P)
    left_cheek_top_left_x = P_x + (0.5 * inter_eye_distance)
    left_cheek_top_left_y = P_y

    # Calculate the coordinates of the left cheek square
    top_left_l = (int(left_cheek_top_left_x), int(left_cheek_top_left_y))
    bottom_right_l = (int(left_cheek_top_left_x + side_length), int(left_cheek_top_left_y + side_length))

    skin_left_cheek = (min(top_left_l[0], bottom_right_l[0]), min(top_left_l[1], bottom_right_l[1]), 
                   max(top_left_l[0], bottom_right_l[0]), max(top_left_l[1], bottom_right_l[1]))

    skin_right_cheek = (min(top_left_r[0], bottom_right_r[0]), min(top_left_r[1], bottom_right_r[1]), 
                        max(top_left_r[0], bottom_right_r[0]), max(top_left_r[1], bottom_right_r[1]))

    IED_margin = int(0.06 * inter_eye_distance)

    _, left_SUN = calculate_EVZ(shape, left_eye_start, 41, IED_margin, 3*IED_margin)
    _, right_SUN = calculate_EVZ(shape, right_eye_start, 47, IED_margin, 3*IED_margin)

    has_makeup_2 = check_makeup(image, left_SUN, right_SUN)
    has_makeup = check_makeup_distance(image, skin_left_cheek, skin_right_cheek, left_SUN, right_SUN)
    return has_makeup or has_makeup_2
        
  

  def light_analysis(self, image_path):
      '''
      Individual check to verify uniform illumination of the face.

      Returns:
        uniform_luminosity: boolean that is true if the image has uniform illumination.
      '''
      image = cv2.imread(image_path)

      image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      detected_boxes = self.detector(image_rgb)
      if not detected_boxes:
         return True
      box = detected_boxes[0]
      
      shape = self.predictor(image_rgb, box)

      uniform_luminosity = self.calculate_luminosity(shape, image_rgb)

      return uniform_luminosity
  
  def mouth_check(self, image_path):
      '''
      Individual check to verify if the mouth is open.

      Returns:
        mouth_open: number that indicates the ratio between the internal distance between the two lips
                    and the height of the lower lip, in accordance to ICAO suggestions.
      '''
      image = cv2.imread(image_path)

      image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      detected_boxes = self.detector(image_rgb)
      if not detected_boxes:
         return True
      box = detected_boxes[0]
      
      shape = self.predictor(image_rgb, box)

      mouth_open = self.calculate_mouth_open(shape)

      return mouth_open
  
  def eyes_open_check(self, image_path):
   '''
    Individual check to verify if the eyes are open

    Returns:
      eyes_open: number that indicates the ratio between the minimum opening between
                 the two eyes and the inter-eye distance in accordance to NIST suggestions.
    '''
     
   image = cv2.imread(image_path)

   image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

   detected_boxes = self.detector(image_rgb)
   box = detected_boxes[0]
   
   shape = self.predictor(image_rgb, box)

   eyes_open = self.calculate_eyes_open(shape)

   return eyes_open
  
  def head_location(self, image_path):
   '''
   computes the middle point of the face, as the middle point between the two left centers.
   From that, you can calculate the ratio suggested as best practice in the ICAO document,
   Mh, which is the distance from the left border, and Mv, the distance from the top border.

   returns:
    m_h and m_v
   '''
   image = cv2.imread(image_path)

   image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

   detected_boxes = self.detector(image_rgb)
   box = detected_boxes[0]
    
   shape = self.predictor(image_rgb, box)

   left_eye_start = 36
   left_eye_end = 39
   right_eye_start = 42
   right_eye_end = 45

   # Calculate middle points of each eye
   left_eye_middle_x = (shape.part(left_eye_start).x + shape.part(left_eye_end).x) / 2
   left_eye_middle_y = (shape.part(left_eye_start).y + shape.part(left_eye_end).y) / 2

   right_eye_middle_x = (shape.part(right_eye_start).x + shape.part(right_eye_end).x) / 2
   right_eye_middle_y = (shape.part(right_eye_start).y + shape.part(right_eye_end).y) / 2

   middle_x = (left_eye_middle_x + right_eye_middle_x) / 2
   middle_y = (left_eye_middle_y + right_eye_middle_y) / 2
   image_height, image_width, _ = image.shape

   # Calculate m_h
   m_h = middle_x

   # Calculate m_v
   m_v = middle_y

   return m_h, m_v, image_width, image_height

  def landmark_analysis(self, image_path):
    '''
    Executes all the controls and returns values for all of them.
    '''
    image = cv2.imread(image_path)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detected_boxes = self.detector(image_rgb)

    if len(detected_boxes) == 0:
       return -1, -1, -1, -1, -1, False, -1, -1 ,True

    box = detected_boxes[0]
    
    shape = self.predictor(image_rgb, box)

    '''
    CALCULATION OF IED
    '''

    left_eye_start = 36
    left_eye_end = 39
    right_eye_start = 42
    right_eye_end = 45

    # Calculate middle points of each eye
    left_eye_middle_x = (shape.part(left_eye_start).x + shape.part(left_eye_end).x) / 2
    left_eye_middle_y = (shape.part(left_eye_start).y + shape.part(left_eye_end).y) / 2

    right_eye_middle_x = (shape.part(right_eye_start).x + shape.part(right_eye_end).x) / 2
    right_eye_middle_y = (shape.part(right_eye_start).y + shape.part(right_eye_end).y) / 2

    middle_x = (left_eye_middle_x + right_eye_middle_x) / 2
    middle_y = (left_eye_middle_y + right_eye_middle_y) / 2


    # Calculate inter-eye distance (IED) between middle points
    inter_eye_distance = math_distance(left_eye_middle_x, left_eye_middle_y, right_eye_middle_x, right_eye_middle_y)

    
    
    IED_margin = int(0.06 * inter_eye_distance)

    left_EVZ, left_SUN = calculate_EVZ(shape, left_eye_start, 41, IED_margin, 3*IED_margin)
    right_EVZ, right_SUN = calculate_EVZ(shape, right_eye_start, 47, IED_margin, 3*IED_margin)

    #left_red_eye = check_red_eye(image, left_EVZ)
    #right_red_eye = check_red_eye(image, right_EVZ)
    
    '''
    CALCULATION OF EYES-OPEN (1 IN THE PAPER)
    '''

    left_eye_opening = max(point_distance(shape.part(37), shape.part(41)),
                           point_distance(shape.part(38), shape.part(40)))
    right_eye_opening = max(point_distance(shape.part(43), shape.part(47)),
                            point_distance(shape.part(44), shape.part(46)))

    # Calculate "eyes open" ratio
    eyes_open = min(left_eye_opening, right_eye_opening) / inter_eye_distance


    '''
    CALCULATION OF MOUTH-OPEN (1 IN THE PAPER)
    '''

    mouth_up = shape.part(62)
    mouth_down = shape.part(66)
    
    # Calculate the midpoint
    mouth_mid_x = (mouth_up.x + mouth_down.x) / 2
    mouth_mid_y = (mouth_up.y + mouth_down.y) / 2

    # Midpoint as a tuple
    mouth_mid = (mouth_mid_x, mouth_mid_y)

    mouth_opening = point_distance(mouth_up, mouth_down)
    lower_lip = point_distance(shape.part(66), shape.part(57))
    mouth_open = 1 - min(1, mouth_opening / lower_lip)
    
    '''
    CALCULATION OF m_h AND m_v
    '''
    image_height, image_width, _ = image.shape
    
    # Calculate m_h
    m_h = middle_x / image_width
    
    # Calculate m_v
    m_v = middle_y / image_height

    uniform_luminosity = self.calculate_luminosity(shape, image_rgb)

    '''
    MAKE UP CHECK
    '''

    emd = math_distance(middle_x, middle_y, mouth_mid_x, mouth_mid_y)

    P_x = middle_x
    P_y = middle_y + (emd / 2)

    side_length = 0.3 * inter_eye_distance

    # Calculate the top-right corner for the right cheek square (0.5 * IED to the left from P)
    right_cheek_top_right_x = P_x - (0.5 * inter_eye_distance)
    right_cheek_top_right_y = P_y

    # Calculate the coordinates of the right cheek square
    top_left_r = (int(right_cheek_top_right_x - side_length), int(right_cheek_top_right_y))
    bottom_right_r = (int(right_cheek_top_right_x), int(right_cheek_top_right_y + side_length))

    # Calculate the top-left corner for the left cheek square (0.5 * IED to the right from P)
    left_cheek_top_left_x = P_x + (0.5 * inter_eye_distance)
    left_cheek_top_left_y = P_y

    # Calculate the coordinates of the left cheek square
    top_left_l = (int(left_cheek_top_left_x), int(left_cheek_top_left_y))
    bottom_right_l = (int(left_cheek_top_left_x + side_length), int(left_cheek_top_left_y + side_length))

    skin_left_cheek = (min(top_left_l[0], bottom_right_l[0]), min(top_left_l[1], bottom_right_l[1]), 
                   max(top_left_l[0], bottom_right_l[0]), max(top_left_l[1], bottom_right_l[1]))

    skin_right_cheek = (min(top_left_r[0], bottom_right_r[0]), min(top_left_r[1], bottom_right_r[1]), 
                        max(top_left_r[0], bottom_right_r[0]), max(top_left_r[1], bottom_right_r[1]))

    makeup_check_1 = check_makeup(image, left_SUN, right_SUN)
    makeup_check_2 = check_makeup_distance(image, skin_left_cheek, skin_right_cheek, left_SUN, right_SUN)

    has_makeup = makeup_check_1 or makeup_check_2

    return inter_eye_distance, eyes_open, mouth_open, m_h, m_v, uniform_luminosity, image_height, image_width, has_makeup
  

  def landmark_list_analysis(self, image_path, checks=None):
    '''
    Executes specified controls and returns values for selected checks.
    If no checks are provided, all controls are executed.
    '''
    if checks is None:
        checks = ['inter_eye_distance', 'eyes_open', 'mouth_open', 'm_h', 'm_v', 
                  'uniform_luminosity', 'has_makeup']
    
    results = {}

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detected_boxes = self.detector(image_rgb)
    box = detected_boxes[0]
    
    shape = self.predictor(image_rgb, box)
    
    # Common calculations used by multiple checks
    left_eye_start = 36
    left_eye_end = 39
    right_eye_start = 42
    right_eye_end = 45

    left_eye_middle_x = (shape.part(left_eye_start).x + shape.part(left_eye_end).x) / 2
    left_eye_middle_y = (shape.part(left_eye_start).y + shape.part(left_eye_end).y) / 2
    right_eye_middle_x = (shape.part(right_eye_start).x + shape.part(right_eye_end).x) / 2
    right_eye_middle_y = (shape.part(right_eye_start).y + shape.part(right_eye_end).y) / 2

    middle_x = (left_eye_middle_x + right_eye_middle_x) / 2
    middle_y = (left_eye_middle_y + right_eye_middle_y) / 2
    
    inter_eye_distance = math_distance(left_eye_middle_x, left_eye_middle_y, 
                                       right_eye_middle_x, right_eye_middle_y)

    # Run checks based on user input
    if 'inter_eye_distance' in checks:
        results['inter_eye_distance'] = inter_eye_distance

    if 'eyes_open' in checks:
        left_eye_opening = max(point_distance(shape.part(37), shape.part(41)),
                               point_distance(shape.part(38), shape.part(40)))
        right_eye_opening = max(point_distance(shape.part(43), shape.part(47)),
                                point_distance(shape.part(44), shape.part(46)))
        eyes_open = min(left_eye_opening, right_eye_opening) / inter_eye_distance
        results['eyes_open'] = eyes_open >= config.EYES_THRESHOLD 

    if 'mouth_open' in checks:
        mouth_up = shape.part(62)
        mouth_down = shape.part(66)
        mouth_opening = point_distance(mouth_up, mouth_down)
        lower_lip = point_distance(shape.part(66), shape.part(57))
        mouth_open = 1 - min(1, mouth_opening / lower_lip)
        results['mouth_open'] = mouth_open >= config.MOUTH_THRESHOLD

    if 'm_h' in checks:
        image_height, image_width, _ = image.shape
        m_h = middle_x / image_width
        results['m_h'] = m_h

    if 'm_v' in checks:
        image_height, image_width, _ = image.shape
        m_v = middle_y / image_height
        results['m_v'] = m_v

    if 'uniform_luminosity' in checks:
        uniform_luminosity = self.calculate_luminosity(shape, image_rgb)
        results['uniform_luminosity'] = uniform_luminosity

    if 'has_makeup' in checks:
        has_makeup = self.calculate_makeup(shape, image)
        results['has_makeup'] = has_makeup

    return results



  
  def detect_and_draw_landmarks(self, image_path, output_path=None):
    '''
    Utility function to visualize the efficacy of the landmark detection
    '''
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detected_boxes = self.detector(image_rgb)

    for box in detected_boxes:
      shape = self.predictor(image_rgb, box)

      # Process the detection boxes
      res_box = process_boxes(box)
      cv2.rectangle(image, (res_box[0], res_box[1]),
                    (res_box[2], res_box[3]), (0, 255, 0),
                    2)
      # Iterate over all keypoints
      for i in range(68): 
        # Draw the keypoints on the detected faces
        cv2.circle(image, (shape.part(i).x, shape.part(i).y),
                  2, (0, 255, 0), -1)
      
    filename, ext = os.path.splitext(os.path.basename(image_path))  # Extract filename and extension
    output_filename = filename + "_landmark" + ext

    if output_path:
      try:
        # Create the output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        # Save with rectangle in the provided output_path (directory)
        faces_dir = output_path
      except OSError as e:
        print(f"Error creating output directory: {e}")
        # Handle the error gracefully, potentially use default saving behavior
    else:
      # Save with rectangle in "faces_detected" folder in script's directory
      script_dir = os.path.dirname(os.path.realpath(__file__))  # Get script's directory
      faces_dir = os.path.join(script_dir, "landmarks_detected")
      os.makedirs(faces_dir, exist_ok=True)  # Create "faces_detected" if it doesn't exist
      
    output_image_path = os.path.join(faces_dir, output_filename)
    cv2.imwrite(output_image_path, image)






    '''
    CALCULATION HELPER

    Below, a set of functions that help compute the individual checks
    '''





  def calculate_IED(self, shape):
    '''
    CALCULATION OF IED
    '''

    left_eye_start = 36
    left_eye_end = 39
    right_eye_start = 42
    right_eye_end = 45

    # Calculate middle points of each eye
    left_eye_middle_x = (shape.part(left_eye_start).x + shape.part(left_eye_end).x) / 2
    left_eye_middle_y = (shape.part(left_eye_start).y + shape.part(left_eye_end).y) / 2

    right_eye_middle_x = (shape.part(right_eye_start).x + shape.part(right_eye_end).x) / 2
    right_eye_middle_y = (shape.part(right_eye_start).y + shape.part(right_eye_end).y) / 2

    # Calculate inter-eye distance (IED) between middle points
    inter_eye_distance = math_distance(left_eye_middle_x, left_eye_middle_y, right_eye_middle_x, right_eye_middle_y)

    return inter_eye_distance
  
  def calculate_eyes_open(self, shape):
   inter_eye_distance = self.calculate_IED(shape)

   left_eye_opening = max(point_distance(shape.part(37), shape.part(41)),
                        point_distance(shape.part(38), shape.part(40)))
   right_eye_opening = max(point_distance(shape.part(43), shape.part(47)),
                           point_distance(shape.part(44), shape.part(46)))

   # Calculate "eyes open" ratio
   eyes_open = min(left_eye_opening, right_eye_opening) / inter_eye_distance

   return eyes_open
  
  def calculate_makeup(self, shape, image):
    '''
    Individual check to verify the presence of excessive makeup.

    Returns:
      a boolean that is true if one of the two makeup checks results positive.
    '''

    left_eye_start = 36
    left_eye_end = 39
    right_eye_start = 42
    right_eye_end = 45

    # Calculate middle points of each eye
    left_eye_middle_x = (shape.part(left_eye_start).x + shape.part(left_eye_end).x) / 2
    left_eye_middle_y = (shape.part(left_eye_start).y + shape.part(left_eye_end).y) / 2

    right_eye_middle_x = (shape.part(right_eye_start).x + shape.part(right_eye_end).x) / 2
    right_eye_middle_y = (shape.part(right_eye_start).y + shape.part(right_eye_end).y) / 2

    middle_x = (left_eye_middle_x + right_eye_middle_x) / 2
    middle_y = (left_eye_middle_y + right_eye_middle_y) / 2


    # Calculate inter-eye distance (IED) between middle points
    inter_eye_distance = math_distance(left_eye_middle_x, left_eye_middle_y, right_eye_middle_x, right_eye_middle_y)
        
    mouth_up = shape.part(62)
    mouth_down = shape.part(66)
        
    # Calculate the midpoint
    mouth_mid_x = (mouth_up.x + mouth_down.x) / 2
    mouth_mid_y = (mouth_up.y + mouth_down.y) / 2
    side_length = 0.3 * inter_eye_distance

    '''
    CALCULATION OF LUMINOSITY
    '''

    emd = math_distance(middle_x, middle_y, mouth_mid_x, mouth_mid_y)

    '''CHEEK SQUARES'''

    P_x = middle_x
    P_y = middle_y + (emd / 2)

    # Calculate the top-right corner for the right cheek square (0.5 * IED to the left from P)
    right_cheek_top_right_x = P_x - (0.5 * inter_eye_distance)
    right_cheek_top_right_y = P_y

    # Calculate the coordinates of the right cheek square
    top_left_r = (int(right_cheek_top_right_x - side_length), int(right_cheek_top_right_y))
    bottom_right_r = (int(right_cheek_top_right_x), int(right_cheek_top_right_y + side_length))

    # Calculate the top-left corner for the left cheek square (0.5 * IED to the right from P)
    left_cheek_top_left_x = P_x + (0.5 * inter_eye_distance)
    left_cheek_top_left_y = P_y

    # Calculate the coordinates of the left cheek square
    top_left_l = (int(left_cheek_top_left_x), int(left_cheek_top_left_y))
    bottom_right_l = (int(left_cheek_top_left_x + side_length), int(left_cheek_top_left_y + side_length))

    skin_left_cheek = (min(top_left_l[0], bottom_right_l[0]), min(top_left_l[1], bottom_right_l[1]), 
                   max(top_left_l[0], bottom_right_l[0]), max(top_left_l[1], bottom_right_l[1]))

    skin_right_cheek = (min(top_left_r[0], bottom_right_r[0]), min(top_left_r[1], bottom_right_r[1]), 
                        max(top_left_r[0], bottom_right_r[0]), max(top_left_r[1], bottom_right_r[1]))

    IED_margin = int(0.06 * inter_eye_distance)

    _, left_SUN = calculate_EVZ(shape, left_eye_start, 41, IED_margin, 3*IED_margin)
    _, right_SUN = calculate_EVZ(shape, right_eye_start, 47, IED_margin, 3*IED_margin)

    has_makeup_2 = check_makeup(image, left_SUN, right_SUN)
    has_makeup = check_makeup_distance(image, skin_left_cheek, skin_right_cheek, left_SUN, right_SUN)
    return has_makeup or has_makeup_2
  


  def calculate_luminosity(self, shape, image_rgb):
    '''
      Individual check to verify the uniform illumination in the face region according to the ICAO standard.

      Returns:
        uniform_luminosity_squares: a boolean that is true if the face is evenly illuminated.
    '''

    left_eye_start = 36
    left_eye_end = 39
    right_eye_start = 42
    right_eye_end = 45

    # Calculate middle points of each eye
    left_eye_middle_x = (shape.part(left_eye_start).x + shape.part(left_eye_end).x) / 2
    left_eye_middle_y = (shape.part(left_eye_start).y + shape.part(left_eye_end).y) / 2

    right_eye_middle_x = (shape.part(right_eye_start).x + shape.part(right_eye_end).x) / 2
    right_eye_middle_y = (shape.part(right_eye_start).y + shape.part(right_eye_end).y) / 2

    middle_x = (left_eye_middle_x + right_eye_middle_x) / 2
    middle_y = (left_eye_middle_y + right_eye_middle_y) / 2


    # Calculate inter-eye distance (IED) between middle points
    inter_eye_distance = math_distance(left_eye_middle_x, left_eye_middle_y, right_eye_middle_x, right_eye_middle_y)
    
    mouth_up = shape.part(62)
    mouth_down = shape.part(66)
    
    # Calculate the midpoint
    mouth_mid_x = (mouth_up.x + mouth_down.x) / 2
    mouth_mid_y = (mouth_up.y + mouth_down.y) / 2



    '''
    CALCULATION OF LUMINOSITY
    '''

    emd = math_distance(middle_x, middle_y, mouth_mid_x, mouth_mid_y)

    ''' SQUARE F (FOREHEAD) '''

    # Calculate the point P (emd / 2 up from M)
    P_x = middle_x
    P_y = middle_y - (emd / 2)

    # Calculate the side length of the square
    side_length = 0.3 * inter_eye_distance

    # Calculate the bottom-left corner of the square
    bottom_left_x = P_x - (0.15 * inter_eye_distance)
    bottom_left_y = P_y

    # Draw the square
    top_left_f = (int(bottom_left_x), int(bottom_left_y - side_length))
    bottom_right_f = (int(bottom_left_x + side_length), int(bottom_left_y))

    ''' SQUARE C (CHIN)'''

    # Calculate the point P' (emd / 2 down from mouth_mid)
    chin_x = mouth_mid_x
    chin_y = mouth_mid_y + (emd / 2)

    # Calculate the top-left corner of the square
    top_left_x = chin_x - (0.15 * inter_eye_distance)
    top_left_y = chin_y - (0.15 * inter_eye_distance)

    # Calculate the coordinates of the square
    top_left_c = (int(top_left_x), int(top_left_y))
    bottom_right_c = (int(top_left_x + side_length), int(top_left_y + side_length))
    
    '''CHEEK SQUARES'''

    P_x = middle_x
    P_y = middle_y + (emd / 2)

    # Calculate the top-right corner for the right cheek square (0.5 * IED to the left from P)
    right_cheek_top_right_x = P_x - (0.5 * inter_eye_distance)
    right_cheek_top_right_y = P_y

    # Calculate the coordinates of the right cheek square
    top_left_r = (int(right_cheek_top_right_x - side_length), int(right_cheek_top_right_y))
    bottom_right_r = (int(right_cheek_top_right_x), int(right_cheek_top_right_y + side_length))

    # Calculate the top-left corner for the left cheek square (0.5 * IED to the right from P)
    left_cheek_top_left_x = P_x + (0.5 * inter_eye_distance)
    left_cheek_top_left_y = P_y

    # Calculate the coordinates of the left cheek square
    top_left_l = (int(left_cheek_top_left_x), int(left_cheek_top_left_y))
    bottom_right_l = (int(left_cheek_top_left_x + side_length), int(left_cheek_top_left_y + side_length))

    mi_forehead = calculate_mean_intensity(image_rgb, top_left_f, bottom_right_f)
    mi_chin = calculate_mean_intensity(image_rgb, top_left_c, bottom_right_c)
    mi_right_cheek = calculate_mean_intensity(image_rgb, top_left_r, bottom_right_r)
    mi_left_cheek = calculate_mean_intensity(image_rgb, top_left_l, bottom_right_l)

    # Collect mean intensities in a dictionary for each channel
    mi_squares = {
        "forehead": mi_forehead,
        "chin": mi_chin,
        "right_cheek": mi_right_cheek,
        "left_cheek": mi_left_cheek
    }

    uniform_luminosity_squares = True

    
    avg = [0, 0, 0]

    for channel in range(3):  # Iterate over RGB channels
        mi_values = [mi_squares[square][channel] for square in mi_squares]
        max_mi = max(mi_values)
        min_mi = min(mi_values)

        avg[channel] =  sum(mi_values) / len(mi_values)

        # Check the condition for each channel
        if min_mi < config.MIN_COLOR_RATIO_THRESOLD * max_mi:
            uniform_luminosity_squares = False
            continue

    if all(a > 220 for a in avg):
      uniform_luminosity_squares = False
    
    return uniform_luminosity_squares
  
  def calculate_mouth_open(self, shape):
    mouth_up = shape.part(62)
    mouth_down = shape.part(66)
    
    # Calculate the midpoint
    mouth_mid_x = (mouth_up.x + mouth_down.x) / 2
    mouth_mid_y = (mouth_up.y + mouth_down.y) / 2

    # Midpoint as a tuple
    mouth_mid = (mouth_mid_x, mouth_mid_y)

    mouth_opening = point_distance(mouth_up, mouth_down)
    lower_lip = point_distance(shape.part(66), shape.part(57))
    mouth_open = 1 - min(1, mouth_opening / lower_lip)

    return mouth_open