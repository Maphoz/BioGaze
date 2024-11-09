import argparse
import glob
import os
import config
from tabulate import tabulate
import time

from detectors import detect
from landmarks import landmark
from head_pose import headpose
from face_parser import parserModel
from emotion_recognizer import emotion_detector
from image_quality import qualitychecker
from gaze_estimation import gaze_estimator

def get_image_paths(input_path):
  """
  Gets a list of image file paths from the given input path (image or directory).

  Args:
    input_path (str): Path to an image or directory.

  Returns:
    list: List of image file paths.
  """
  if os.path.isfile(input_path):
    return [input_path]
  elif os.path.isdir(input_path):
    # Use glob to search for image extensions (modify pattern if needed)
    image_paths = glob.glob(os.path.join(input_path, "*.jpg"))
    image_paths.extend(glob.glob(os.path.join(input_path, "*.jpeg")))
    image_paths.extend(glob.glob(os.path.join(input_path, "*.png")))
    return image_paths
  else:
    raise ValueError(f"Invalid input path: {input_path}")
  
def write_filename_to_file(filename, output_file):
    """
    Write the filename to the output file.

    Args:
      filename (str): Name of the file.
      output_file (str): Path to the output file.
    """
    with open(output_file, 'a') as f:
        f.write(f"\nImage analysed: {filename}\n\n")

def write_rejection(output_file, rejection_message):
    """
    Write the filename to the output file.

    Args:
      filename (str): Name of the file.
      output_file (str): Path to the output file.
    """
    with open(output_file, 'a') as f:
        f.write(f"Image rejected: {rejection_message}\n\n")

def write_approved(output_file):
    """
    Write the filename to the output file.

    Args:
      filename (str): Name of the file.
      output_file (str): Path to the output file.
    """
    with open(output_file, 'a') as f:
        f.write(f"Image approved.\n\n")

def write_faces_detected_to_file(n_faces, output_file):
    """
    Write the filename to the output file.

    Args:
      filename (str): Name of the file.
      output_file (str): Path to the output file.
    """
    with open(output_file, 'a') as f:
        f.write(f"Faces detected: {n_faces}\n\n")

def write_headpose_to_file(pitch, yaw, roll, output_file):
    """
    Write the filename to the output file.

    Args:
      filename (str): Name of the file.
      output_file (str): Path to the output file.
    """
    pitch_v = round(pitch, 2)
    yaw_v = round(yaw, 2)
    roll_v = round(roll, 2)

    with open(output_file, 'a') as f:
        f.write(f"Pitch: {pitch_v}\nYaw: {yaw_v}\nRoll: {roll_v}\n\n")

def write_summary_to_file(total_images, images_approved, images_rejected, output_file):
    """
    Write summary information (total images, approved images, rejected images) to the output file.

    Args:
      total_images (int): Total number of images processed.
      images_approved (int): Number of images approved.
      images_rejected (int): Number of images rejected.
      output_file (str): Path to the output file.
    """
    with open(output_file, 'r+') as f:
        content = f.read()
        f.seek(0, 0)  # Move cursor to the beginning of the file
        f.write(f"\nTotal Images: {total_images}\n")
        f.write(f"Approved Images: {images_approved}\n")
        f.write(f"Rejected Images: {images_rejected}\n\n")
        f.write(content)  # Write back the original content after the summary

def write_landmark_to_file(inter_eye_distance, eyes_open, mouth_open, m_h, m_v, output_file):
    
    ied = round(inter_eye_distance, 2)
    e_open = round(eyes_open, 2)
    m_open = round(mouth_open, 2)
    hor_cen = round(m_h, 2)
    ver_cen = round(m_v, 2)

    with open(output_file, 'a') as f:
        f.write(f"IED: {ied}\n")
        f.write(f"Eyes open: {e_open}\n")
        f.write(f"Mouth open: {m_open}\n")
        f.write(f"Centratura orizzontale: {hor_cen}\n")
        f.write(f"Centratura verticale: {ver_cen}\n\n")



def main():
  parser = argparse.ArgumentParser(description="Perform 14 face image quality checks in accordance to ISO/ICAO standards.")
  parser.add_argument(
      "-i",
      "--input",
      type=str,
      required=True,
      help="Path to an image or directory containing images"
  )
  parser.add_argument(
      "-o",
      "--output",
      type=str,
      default=None,
      help="Output path to save report (optional)",
  )

  try:
    args = parser.parse_args()
    table_output = "table_results.txt"

    if args.output is None:
      output_file = "verbose_results.txt"
    else:
      output_file = args.output + ".txt"

    detector = detect.FaceDetector()  # Initialize detector
    landmark_recognizer = landmark.LandmarkRecognizer()
    pose_estimator = headpose.HeadposeEstimator()
    face_parser = parserModel.FaceParser()
    emotion_recognizer = emotion_detector.EmotionDetector()
    quality_checker = qualitychecker.QualityChecker()
    gaze_model = gaze_estimator.GazeEstimator()


    total_images = 0
    images_approved = 0
    images_rejected = 0
    table_data = []

    #process the directory
    for image_path in get_image_paths(args.input):

      #keep a list with the values of the current image being processed
      image_data = []

      has_been_rejected = False
      filename = os.path.basename(image_path)
      write_filename_to_file(filename, output_file)
      total_images += 1


      #start with face detection. If there isn't just one face, interrupt the flow of the code
      #the image is not compliant
      faces_detected, correct_exposure = detector.detector_analysis(image_path)

      write_faces_detected_to_file(faces_detected, output_file)

      if faces_detected != config.MAX_FACES:
        images_rejected += 1
        if faces_detected > config.MAX_FACES:
          write_rejection(output_file, "more than one face detected.")
        if faces_detected == 0:
          write_rejection(output_file, "no face detected.")
        
        #write a table without values, no control was possible
        image_data = [filename, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        table_data.append(image_data)
        continue

      
      if not correct_exposure:
        if not has_been_rejected:
           images_rejected += 1
           has_been_rejected = True
        write_rejection(output_file, "Exposure not compliant.")

      #progress with the head pose estimation
      pitch, yaw, roll = pose_estimator.get_headpose_values(image_path)

      pitch_out = round((pitch + 90) / 180, 2)
      yaw_out = round((yaw + 90) / 180, 2)
      roll_out = round((roll + 90) / 180, 2)

      #keep a boolean for the three values of the head pose
      frontal_pose = True   

      write_headpose_to_file(pitch, yaw, roll, output_file)

      if yaw < config.MIN_YAW or yaw > config.MAX_YAW:
        if not has_been_rejected:
           images_rejected += 1
           has_been_rejected = True

        frontal_pose = False
        write_rejection(output_file, "excessive yaw detected.")

      if pitch < config.MIN_PITCH or pitch > config.MAX_PITCH:
        if not has_been_rejected:
           images_rejected += 1
           has_been_rejected = True

        frontal_pose = False
        write_rejection(output_file, "excessive pitch detected.")

      if roll < config.MIN_ROLL or roll > config.MAX_ROLL:
        if not has_been_rejected:
           images_rejected += 1
           has_been_rejected = True

        frontal_pose = False
        write_rejection(output_file, "excessive roll detected.")


      #now lets perform the controls related to the face parser
      has_hat, color_saturation, has_glasses, head_not_contained, chin_not_contained, shoulder_check, uniform_illumination, homogeneous_background, has_sunglasses = face_parser.parser_analysis(image_path)
      
      if has_hat:
        if not has_been_rejected:
           images_rejected += 1
           has_been_rejected = True
        write_rejection(output_file, "person is wearing a hat.")
      
      if not homogeneous_background:
        if not has_been_rejected:
           images_rejected += 1
           has_been_rejected = True
        write_rejection(output_file, "Background not homogeneous.")

      if not shoulder_check:
        if not has_been_rejected:
          images_rejected += 1
          has_been_rejected = True
        frontal_pose = False
        write_rejection(output_file, "Shoulders not aligned.") 

      if color_saturation:
        if not has_been_rejected:
          images_rejected += 1
          has_been_rejected = True
        write_rejection(output_file, "Color saturation not compliant.")

      if chin_not_contained or head_not_contained:
        if not has_been_rejected:
          images_rejected += 1
          has_been_rejected = True
        write_rejection(output_file, "Crown or chin not contained.")
         
      
      inter_eye_distance, eyes_open, mouth_open, m_h, m_v, uniform_luminosity, image_height, image_width, has_makeup = landmark_recognizer.landmark_analysis(image_path)
      
      #image_ratio = round(image_width / image_height, 2)
      #head_location = True
      #head_dimensions = True
      
      write_landmark_to_file(inter_eye_distance, eyes_open, mouth_open, m_h, m_v, output_file)

      
      #if min(1, inter_eye_distance / 180) < config.MINIMUM_IED:
      #   images_rejected += 1
      
      eyes_open_compliant = True
      if eyes_open < config.EYES_THRESHOLD:
        eyes_open_compliant = False
        if not has_been_rejected:
          images_rejected += 1
          has_been_rejected = True
        write_rejection(output_file, "eyes closed detected.")

      if mouth_open < config.MOUTH_THRESHOLD:
        if not has_been_rejected:
          images_rejected += 1
          has_been_rejected = True
        write_rejection(output_file, "mouth open detected.")

      if has_glasses and has_sunglasses:
        if not has_been_rejected:
          images_rejected += 1
          has_been_rejected = True
        write_rejection(output_file, "sunglasses detected.")
      
      if has_makeup:
        if not has_been_rejected:
          images_rejected += 1
          has_been_rejected = True
        write_rejection(output_file, "Excessive makeup detected.")

      if not uniform_illumination or not uniform_luminosity:
        if not has_been_rejected:
          images_rejected += 1
          has_been_rejected = True
        write_rejection(output_file, "face ilummination not uniform.")
        

      #CONTROLS MOMENTARELY SUSPENDED
      '''
      if image_ratio < config.MIN_WIDTH_HEIGHT_RATIO or roll > config.MAX_WIDTH_HEIGHT_RATIO:
        if not has_been_rejected:
           images_rejected += 1
           has_been_rejected = True
        write_rejection(output_file, "image dimensions not compliant.")
      


      if (m_h / image_width) < config.MIN_LEFT_DISTANCE or (m_h / image_width) > config.MAX_LEFT_DISTANCE or (m_v / image_height) < config.MIN_TOP_DISTANCE or (m_v / image_height) > config.MAX_TOP_DISTANCE:
        head_location = False
        if not has_been_rejected:
           images_rejected += 1
           has_been_rejected = True
        write_rejection(output_file, "head position not compliant.")
      
      if (head_width / image_width) < config.MIN_WIDTH_HEAD or (head_width / image_width) > config.MAX_WIDTH_HEAD or (head_height / image_height) < config.MIN_HEIGHT_HEAD or (head_height / image_height) > config.MAX_HEIGHT_HEAD:
        head_dimensions = False
        if not has_been_rejected:
           images_rejected += 1
           has_been_rejected = True
        write_rejection(output_file, "head dimensions not compliant.")
      '''


      #Emotion recognition control
      neutral_expression = emotion_recognizer.check_neutral_expression(image_path)

      if not neutral_expression:
        if not has_been_rejected:
          images_rejected += 1
          has_been_rejected = True
        write_rejection(output_file, "Non neutral emotion detected.")

      gaze_in_camera = gaze_model.calculate_gaze(image_path)

      if not gaze_in_camera:
        if not has_been_rejected:
          images_rejected += 1
          has_been_rejected = True
        write_rejection(output_file, "Gaze non directed towards camera.")


      #Computer vision checks

      is_posterized = quality_checker.is_posterized(image_path)
      is_pixelated = quality_checker.is_pixelated(image_path)
      out_of_focus = quality_checker.is_out_of_focus(image_path)

      if is_posterized:
        if not has_been_rejected:
          images_rejected += 1
          has_been_rejected = True
        write_rejection(output_file, "Posterization effect detected.")

      if is_pixelated:
        if not has_been_rejected:
          images_rejected += 1
          has_been_rejected = True
        write_rejection(output_file, "Excessive pixelation detected.")


      if out_of_focus:
        if not has_been_rejected:
          images_rejected += 1
          has_been_rejected = True
        write_rejection(output_file, "Image is out of focus.")


      if not has_been_rejected:
        images_approved += 1
        write_approved(output_file)


      #image_data = [filename, inter_eye_distance, is_homogeneous, uniform_luminosity, color_saturation, roll_out, pitch_out, yaw_out, shoulder_check, mouth_open, eye_quality, has_glasses, has_hat, image_ratio, head_location, head_dimensions]
      image_data = [filename, not has_been_rejected, not has_hat, eyes_open_compliant, not (has_glasses and has_sunglasses), not is_posterized, gaze_in_camera, neutral_expression, not out_of_focus, correct_exposure, not has_makeup, not is_pixelated, frontal_pose, not color_saturation, homogeneous_background, uniform_illumination and uniform_luminosity, inter_eye_distance, roll_out, pitch_out, yaw_out, mouth_open, eyes_open, has_glasses]
      table_data.append(image_data)


    write_summary_to_file(total_images, images_approved, images_rejected, output_file)
    table = tabulate(table_data, config.HEADERS_TABLE, tablefmt='grid')

    # Write the table to the output file
    with open(table_output, 'w') as f:
        f.write(table)


  except argparse.ArgumentError as e:
    parser.error(f"{e}\nInput file or directory path must be provided.")


if __name__ == "__main__":
  main()
