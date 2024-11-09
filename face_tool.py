import argparse
import glob
import os
#from pose import pose  # (Import only if needed)

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


def main():
  parser = argparse.ArgumentParser(description="Face Processing App")
  parser.add_argument(
      "-i",
      "--input",
      type=str,
      required=True,
      help="Path to an image or directory containing images"
  )
  parser.add_argument(
      "-d", "--detect", action="store_true", default=False, help="Perform face detection"
  )
  parser.add_argument(
      "-l", "--landmark", action="store_true", default=False, help="Perform landmark recognition"
  )
  parser.add_argument("-p", "--parse", action="store_true", default=False, help="Perform face parsing")
  parser.add_argument(
      "-o",
      "--output",
      type=str,
      default=None,
      help="Output path to save processed images (optional)",
  )

  try:
    args = parser.parse_args()

    # Load modules only if their flags are set
    if args.detect:
      from detectors import detect
      detector = detect.FaceDetector()  # Initialize detector
    else:
      detector = None

    if args.landmark:  # Import only if needed and flag is set
      from landmarks import landmark  # (Import only if needed)
      landmark_recognizer = landmark.LandmarkRecognizer()
    else:
      landmark_recognizer = None

    if args.parse:  # Import only if needed and flag is set
      from face_parser import parserModel
      face_parser = parserModel.FaceParser()
    else:
      face_parser = None

    # Process images (modify logic based on your implementation)
    for image_path in get_image_paths(args.input):
      output_path = args.output
      if detector:
        if output_path:
          detector.detect_and_draw_faces(image_path, output_path=output_path)
        else:
          detector.detect_and_draw_faces(image_path)

      if landmark_recognizer:  # Perform landmark recognition only if faces detected
        if output_path:
          landmark_recognizer.detect_and_draw_landmarks(image_path, output_path=output_path)
        else:
          landmark_recognizer.detect_and_draw_landmarks(image_path)
              
      if face_parser:  # Perform pose estimation only if faces detected
        if output_path:
          face_parser.parse_and_save_faces(image_path, output_path=output_path)
        else:
          face_parser.parse_and_save_faces(image_path)

  except argparse.ArgumentError as e:
    parser.error(f"{e}\nInput file or directory path must be provided.")


if __name__ == "__main__":
  main()
