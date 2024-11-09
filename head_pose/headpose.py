import config
import numpy as np
import cv2
import mediapipe as mp

class HeadposeEstimator:
    def __init__(self):
        """
        Initializes the face detector with the specified model path.
        """
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def get_headpose_values(self, image_path):
        # Load image
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        # Process the image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        results = self.face_mesh.process(image_rgb)

        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_2d = []
        face_3d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [33, 263, 1, 61, 291, 199]:  # Specified landmarks
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

            # Convert lists to numpy arrays
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # Camera matrix
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                   [0, focal_length, img_h / 2],
                                   [0, 0, 1]])
            distortion_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)

            if success:
                # Get rotation matrix and angles
                rmat, jac = cv2.Rodrigues(rotation_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                pitch = angles[0] * 360
                yaw = angles[1] * 360
                roll = angles[2] * 360

                return pitch, yaw, roll

    def headpose_compliant(self, image_path):
        # Load image
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        # Process the image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        results = self.face_mesh.process(image_rgb)

        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_2d = []
        face_3d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [33, 263, 1, 61, 291, 199]:  # Specified landmarks
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

            # Convert lists to numpy arrays
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # Camera matrix
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                   [0, focal_length, img_h / 2],
                                   [0, 0, 1]])
            distortion_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)

            if success:
                # Get rotation matrix and angles
                rmat, jac = cv2.Rodrigues(rotation_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                pitch = angles[0] * 360
                yaw = angles[1] * 360
                roll = angles[2] * 360

                # Check head pose compliance
                if pitch < config.MIN_PITCH or pitch > config.MAX_PITCH:
                    return False
                
                if yaw < config.MIN_YAW or yaw > config.MAX_YAW:
                    return False
                
                if roll < config.MIN_ROLL or roll > config.MAX_ROLL:
                    return False

                return True
