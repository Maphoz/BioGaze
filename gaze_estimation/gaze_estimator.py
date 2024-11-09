import os
import config
import cv2
import dlib
from imutils import face_utils
import numpy as np
import torch
from torchvision import transforms
from . import model
import time

#shape_predictor_path = './modules/shape_predictor_68_face_landmarks.dat'
#face_model_path = 'face_model.txt'
#camera_calibration_path = './example/input/front_cam.xml'
#model_checkpoint_path = './ckpt/epoch_24_ckpt.pth.tar'

class GazeEstimator:
    def __init__(self, shape_predictor_path='./modules/shape_predictor_68_face_landmarks.dat', face_model_path='face_model.txt', camera_calibration_path='./example/input/front_cam.xml', model_checkpoint_path='./ckpt/epoch_24_ckpt.pth.tar'):
        current_dir = os.path.dirname(os.path.abspath(__file__))

        shape_predictor_path = os.path.join(current_dir, shape_predictor_path)

        face_model_path = os.path.join(current_dir, face_model_path)

        camera_calibration_path = os.path.join(current_dir, camera_calibration_path)

        model_checkpoint_path = os.path.join(current_dir, model_checkpoint_path)

        # Initialize face detector and shape predictor
        self.predictor = dlib.shape_predictor(shape_predictor_path)
        self.face_detector = dlib.get_frontal_face_detector()

        # Load camera calibration
        fs = cv2.FileStorage(camera_calibration_path, cv2.FILE_STORAGE_READ)
        self.camera_matrix = fs.getNode('Camera_Matrix').mat()
        self.camera_distortion = fs.getNode('Distortion_Coefficients').mat()

        # Load face model
        face_model_load = np.loadtxt(face_model_path)
        landmark_use = [20, 23, 26, 29, 15, 19]  # Indices for eyes and nose corners
        self.face_model = face_model_load[landmark_use, :]

        # Load the pre-trained gaze model
        self.model = model.gaze_network()
        ckpt = torch.load(model_checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(ckpt['model_state'], strict=True)
        self.model.eval()

        # Image transformation
        self.trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def estimateHeadPose(self, landmarks, face_model, camera, distortion, iterate=True):
        ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)
        if iterate:
            ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)
        return rvec, tvec

    def normalizeData_face_frontcam(self, img, face_model, landmarks, hr, ht, cam):
        focal_norm = 960  # focal length of normalized camera
        distance_norm = 600  # normalized distance between eye and camera
        roiSize = (224, 224)  # size of cropped eye image

        ht = ht.reshape((3, 1))
        hR = cv2.Rodrigues(hr)[0]
        Fc = np.dot(hR, face_model.T) + ht
        two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
        nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
        face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape((3, 1))

        distance = np.linalg.norm(face_center)
        z_scale = distance_norm / distance
        cam_norm = np.array([
            [focal_norm, 0, roiSize[0] / 2],
            [0, focal_norm, roiSize[1] / 2],
            [0, 0, 1.0],
        ])
        S = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, z_scale],
        ])

        forward = (face_center / distance).reshape(3)
        down = np.array([0.0, 1.0, 0.0])
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        down /= np.linalg.norm(down)
        R = np.c_[right, down, forward].T

        W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))

        img_warped = cv2.warpPerspective(img, W, roiSize)
        hR_norm = np.dot(R, hR)
        hr_norm = cv2.Rodrigues(hR_norm)[0]

        num_point = landmarks.shape[0]
        landmarks_warped = cv2.perspectiveTransform(landmarks, W)
        landmarks_warped = landmarks_warped.reshape(num_point, 2)

        return img_warped, landmarks_warped

    def calculate_gaze(self, img_path):
        image = cv2.imread(img_path)
        detected_faces = self.face_detector(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1)

        if len(detected_faces) == 0:
            print("No face detected.")
            return False

        shape = self.predictor(image, detected_faces[0])
        shape = face_utils.shape_to_np(shape)
        landmarks = np.asarray(shape)

        landmarks_sub = landmarks[[36, 39, 42, 45, 31, 35], :].astype(float).reshape(6, 1, 2)
        facePts = self.face_model.reshape(6, 1, 3)

        hr, ht = self.estimateHeadPose(landmarks_sub, facePts, self.camera_matrix, self.camera_distortion)
        img_normalized, _ = self.normalizeData_face_frontcam(image, self.face_model, landmarks_sub, hr, ht, self.camera_matrix)

        input_var = img_normalized[:, :, [2, 1, 0]]  # Convert BGR to RGB
        input_var = self.trans(input_var)
        input_var = torch.autograd.Variable(input_var.float())
        input_var = input_var.view(1, input_var.size(0), input_var.size(1), input_var.size(2))

        pred_gaze = self.model(input_var)[0].detach().numpy()

        hor_look = pred_gaze[1]  # Get yaw component

        # Compliance check
        if hor_look > config.MAXIMUM_LEFT_THRESHOLD or hor_look < config.MAXIMUM_RIGHT_THRESHOLD:
            return False
        else:
            return True
