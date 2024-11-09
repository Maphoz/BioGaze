#DETECTION

MAX_FACES = 1

'''
HEADPOSE THRESHOLDS
'''

MIN_YAW = -5
MAX_YAW = 5

MIN_PITCH = -5
MAX_PITCH = 5

MIN_ROLL = -8
MAX_ROLL = 8


'''
EYES OPEN, MOUTH OPEN THREHSOLDS
'''

MOUTH_THRESHOLD = 0.5
EYES_THRESHOLD = 0.09


'''
IMAGE AND FACE DIMENSION THRESHOLDS
'''
MIN_WIDTH_HEIGHT_RATIO = 0.74
MAX_WIDTH_HEIGHT_RATIO = 0.80

MIN_LEFT_DISTANCE = 0.45
MAX_LEFT_DISTANCE = 0.55

MIN_TOP_DISTANCE = 0.30
MAX_TOP_DISTANCE = 0.50

MIN_WIDTH_HEAD = 0.50
MAX_WIDTH_HEAD = 0.75

MIN_HEIGHT_HEAD = 0.60
MAX_HEIGHT_HEAD = 0.90

MINIMUM_IED = 0.50


#EXPOSURE CONSTANTS

#vecchie
#AVG_DARK_THRESHOLD = 0.0058
#MAX_DARK_THRESHOLD = 0.01

AVG_DARK_THRESHOLD = 0.0055
MAX_DARK_THRESHOLD = 0.015

#vecchie
#AVG_LIGHT_THRESHOLD = 0.0036
#MAX_LIGHT_THRESHOLD = 0.007

AVG_LIGHT_THRESHOLD = 0.0035
MAX_LIGHT_THRESHOLD = 0.015

#PIXELATION THRESHOLDS

PIXELATED_MIN_THRESHOLD = 19
PIXELATED_MAX_THRESHOLD = 106


#EMOTION DETECTION THRESHOLD

MAX_RATIO_EMOTION = 0.35


'''
PARSER THRESHOLDS
'''

#SHOULDER CHECK

MAX_SHOULDER_PIXEL_RATIO = 0.55
MAX_SHOULDER_Y_DISTANCE = 5


#SATURATION CHECK

OVERSATURATION_THRESHOLD = 0.2
UNDERSATURATION_THRESHOLD = 6


#FACE ILLUMINATION

MAX_BRIGHT_LIGHT = 5
MAX_DARK_LIGHT = 70

#BACKGROUND CHECK

MAX_EDGES_THRESHOLD = 900000000
AVG_VARIANCE_THRESHOLD = 500
HOMOGENEOUS_PROPORTION_THRESHOLD = 0.90
SUPERPIXEL_VARIANCE_THRESHOLD = 45

MAX_LIGHT_DARK_SUN = 0.004

'''
COMPUTER VISION THRESHOLDS
'''

#OUT OF FOCUS

MINIMUM_FOCUS_THRESHOLD = 139


#POSTERIZATION

MAX_GAPS_THRESHOLD = 417
GAP_HISTOGRAM_THRESHOLD = 0.001


#PIXELIZATION

PIXEL_SCORE_THRESHOLD = 0.5
MIN_PIXEL_SCORE = 10
MAX_PIXEL_SCORE = 110


'''
LANDMARK THRESHOLDS
'''

#FACE ILLUMINATION

MIN_COLOR_RATIO_THRESOLD = 0.5


#MAKE UP

MAKEUP_HIGH_HUE_THRESHOLD = 0.15

MAKEUP_DISTANCE_THRESHOLD = 14


'''
GAZE ESTIMATION THRESHOLDS
'''

MAXIMUM_RIGHT_THRESHOLD = -0.15
MAXIMUM_LEFT_THRESHOLD = 0.08


'''
HEADERS FOR TABLE OUTPUT
'''

HEADERS = ["Filename", "Compliant", "IED", "Background", "Lighting", "Color/Saturation", "Roll", "Pitch", "Yaw",
           "Shoulders", "Mouth", "Eyes", "Glasses", "Hat", "Image ratio", "Head location", "Head dimension"]

HEADERS_SECONDARY = ["Filename", "Compliant", "Head without covering", "Eyes open", "No sunglasses", "No posterization", 
                     "Gaze in camera", "Neutral expression", "In focus", "Correct exposure", "No/light makeup", "No pixelation", 
                     "Frontal pose", "Correct saturation", "Uniform background", "Uniform face lighting"]


HEADERS_TABLE = ["Filename", "Compliant", "Head without covering", "Eyes open", "No sunglasses", "No posterization", 
                     "Gaze in camera", "Neutral expression", "In focus", "Correct exposure", "No/light makeup", "No pixelation", 
                     "Frontal pose", "Correct saturation", "Uniform background", "Uniform face lighting", "IED", "Roll", "Pitch", "Yaw", "Mouth", "Eyes", "Glasses"]



'''
CONSTANTS FOR INDIVIDUAL CHECKS
'''

# Constants for Checks
HEAD_WITHOUT_COVERING = 0
EYES_OPEN = 1
NO_SUNGLASSES = 2
NO_POSTERIZATION = 3
GAZE_IN_CAMERA = 4
NEUTRAL_EXPRESSION = 5
IN_FOCUS_PHOTO = 6
CORRECT_EXPOSURE = 7
NO_LIGHT_MAKEUP = 8
NO_PIXELATION = 9
FRONTAL_POSE = 10
CORRECT_SATURATION = 11
UNIFORM_BACKGROUND = 12
UNIFORM_FACE_LIGHTING = 13

# Maps of checks to functions for landmark and parser
landmark_checks_map = {
    EYES_OPEN: 'eyes_open',
    NEUTRAL_EXPRESSION: 'mouth_open',
    NO_LIGHT_MAKEUP: 'has_makeup',
    UNIFORM_FACE_LIGHTING: 'uniform_luminosity'
}

parser_checks_map = {
    HEAD_WITHOUT_COVERING: 'has_hat',
    NO_SUNGLASSES: 'has_sunglasses',
    FRONTAL_POSE: 'shoulder_check',
    CORRECT_SATURATION: 'color_saturation',
    UNIFORM_BACKGROUND: 'homogeneous_background',
    UNIFORM_FACE_LIGHTING: 'uniform_illumination'
}

