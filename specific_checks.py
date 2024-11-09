import argparse
import config  # Import your config file with constants

from detectors import detect
from landmarks import landmark
from head_pose import headpose
from face_parser import parserModel
from emotion_recognizer import emotion_detector
from image_quality import qualitychecker
from gaze_estimation import gaze_estimator

# Function to collect the checks and compute the results
def run_checks(image_path, checks, correct_exposure):
    landmark_checks = []
    parser_checks = []
    combined_results = {}

    landmark_recognizer = landmark.LandmarkRecognizer()
    pose_estimator = headpose.HeadposeEstimator()
    face_parser = parserModel.FaceParser()
    emotion_recognizer = emotion_detector.EmotionDetector()
    quality_checker = qualitychecker.QualityChecker()
    gaze_model = gaze_estimator.GazeEstimator()

    # Map checks to landmark and parser
    for check in checks:
        if check in config.landmark_checks_map:
            landmark_checks.append(config.landmark_checks_map[check])
        if check in config.parser_checks_map:
            parser_checks.append(config.parser_checks_map[check])

    # Run the landmark and parser analyses
    landmark_results = landmark_recognizer.landmark_list_analysis(image_path, landmark_checks) if landmark_checks else {}
    parser_results = face_parser.parser_list_analysis(image_path, parser_checks) if parser_checks else {}

    # Combine results based on checks
    for check in checks:
        if check == config.HEAD_WITHOUT_COVERING:
            #the control is has_hat, so we have to negate this
            combined_results['Head_without_covering'] = not parser_results[config.parser_checks_map[check]]
        elif check == config.EYES_OPEN:
            combined_results['Eyes_open'] = landmark_results[config.landmark_checks_map[check]]
        elif check == config.NO_SUNGLASSES:
            combined_results['No_sunglasses'] = not (parser_results[config.parser_checks_map[check]] and landmark_results[config.landmark_checks_map[check]])
        elif check == config.NO_POSTERIZATION:
            combined_results['No_posterization'] = not quality_checker.is_posterized(image_path)
        elif check == config.GAZE_IN_CAMERA:
            combined_results['Gaze_compliant'] = gaze_model.calculate_gaze(image_path)
        elif check == config.NEUTRAL_EXPRESSION:
            combined_results['Neutral_expression'] = emotion_recognizer.check_neutral_expression(image_path) and landmark_results[config.landmark_checks_map[check]]
        elif check == config.IN_FOCUS_PHOTO:
            combined_results['In_focus'] = not quality_checker.is_out_of_focus(image_path)
        elif check == config.CORRECT_EXPOSURE:
            combined_results['Correct_exposure'] = correct_exposure
        elif check == config.NO_LIGHT_MAKEUP:
            combined_results['No_light_makeup'] = not landmark_results[config.landmark_checks_map[check]]
        elif check == config.NO_PIXELATION:
            combined_results['No_pixelation'] = not quality_checker.is_pixelated(image_path)
        elif check == config.FRONTAL_POSE:
            combined_results['Frontal_pose'] = pose_estimator.headpose_compliant(image_path) and parser_results[config.parser_checks_map[check]]
        elif check == config.CORRECT_SATURATION:
            combined_results['Correct_saturation'] = not parser_results[config.parser_checks_map[check]]
        elif check == config.UNIFORM_BACKGROUND:
            combined_results['Uniform_background'] = parser_results[config.parser_checks_map[check]]
        elif check == config.UNIFORM_FACE_LIGHTING:
            combined_results['Uniform_face_lighting'] = parser_results[config.parser_checks_map[check]] and landmark_results[config.landmark_checks_map[check]]

    return combined_results

# Function to print the list of checks and their descriptions
def print_checks_list():
    checks_list = {
        0: "HEAD_WITHOUT_COVERING",
        1: "EYES_OPEN",
        2: "NO_SUNGLASSES",
        3: "NO_POSTERIZATION",
        4: "GAZE_IN_CAMERA",
        5: "NEUTRAL_EXPRESSION",
        6: "IN_FOCUS_PHOTO",
        7: "CORRECT_EXPOSURE",
        8: "NO_LIGHT_MAKEUP",
        9: "NO_PIXELATION",
        10: "FRONTAL_POSE",
        11: "CORRECT_SATURATION",
        12: "UNIFORM_BACKGROUND",
        13: "UNIFORM_FACE_LIGHTING",
    }
    print("Available checks:")
    for check_id, description in checks_list.items():
        print(f"{check_id}: {description}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run face image quality checks on the image.")
    parser.add_argument("-i",
                        "--input",
                        type=str,
                        help="Path to an image"
                        )
    parser.add_argument('-c', '--checks', nargs='+', help="List of integers representing the checks to perform, or 'all' for all checks.")
    parser.add_argument('--list-checks', action='store_true', help="List available checks and their descriptions.")
    return parser.parse_args()


if __name__ == '__main__':
    # Parse command line inputs
    args = parse_arguments()

    # If --list-checks is provided, print the list and exit
    if args.list_checks:
        print_checks_list()

    # Validate inputs and provide error messages
    elif args.input and args.checks:
        image_path = args.input

        # Handle the case where 'all' is specified for checks
        if args.checks == ['all']:
            checks = list(range(14))  # Assumes there are 14 checks, numbered 0-13
        else:
            try:
                checks = list(map(int, args.checks))  # Convert checks to integers
            except ValueError:
                print("Error: Invalid check values. Use integers or 'all' to specify all checks.")
                exit(1)

        # Initialize detector
        detector = detect.FaceDetector()

        faces_detected, correct_exposure = detector.detector_analysis(image_path)

        if faces_detected != config.MAX_FACES:
            print(f"Error: Detected {faces_detected} face(s), but {config.MAX_FACES} face(s) are required for analysis.")
            print("Please ensure that the image contains the correct number of faces and try again.")
            exit(1)

        # Run the checks based on input
        results = run_checks(image_path, checks, correct_exposure)

        print("\nResults of the checks:")
        for check, result in results.items():
            print(f"{check}: {result}")

    # Handle missing inputs with error messages
    elif args.input and not args.checks:
        print("Error: Please specify checks to perform using '-c' when an input image is provided.")
    elif args.checks and not args.input:
        print("Error: Please specify an input image using '-i' when providing checks to perform.")
    else:
        print("Error: No valid arguments provided. Use '-i' for input, '-c' for checks, or '--list-checks' to view available checks.")
