from typing import List

import logging

import numpy as np
import torch
import yacs.config

from gaze_estimation.gaze_estimator.common import Camera, Face, FacePartsName, MODEL3D
from .head_pose_estimation import HeadPoseNormalizer, LandmarkEstimator
from gaze_estimation import (GazeEstimationMethod, create_model,
                             create_transform)

from gaze_estimation.models.blink import BlinkModel
from torchvision import transforms
import cv2

logger = logging.getLogger(__name__)


class GazeEstimator:
    EYE_KEYS = [FacePartsName.REYE, FacePartsName.LEYE]

    def __init__(self, config: yacs.config.CfgNode):
        self._config = config

        self.camera = Camera(config.gaze_estimator.camera_params)
        self._normalized_camera = Camera(
            config.gaze_estimator.normalized_camera_params)

        self._landmark_estimator = LandmarkEstimator(config)
        self._head_pose_normalizer = HeadPoseNormalizer(
            self.camera, self._normalized_camera,
            self._config.gaze_estimator.normalized_camera_distance)
        self._gaze_estimation_model = self._load_model()
        self._transform = create_transform(config)

        # self._blink_model = self._load_blink_model()
        # self._blink_transform = transforms.Compose([transforms.ToTensor()])

    def _load_model(self) -> torch.nn.Module:
        model = create_model(self._config)
        checkpoint = torch.load(self._config.gaze_estimator.checkpoint,
                                map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.to(torch.device(self._config.device))
        model.eval()
        return model
    
    def _load_blink_model(self) -> torch.nn.Module:
        model = BlinkModel(num_classes=2)
        checkpoint = torch.load("data/models/blink/model_11_96_0.1256.t7", 
                                map_location='cpu')
        model.load_state_dict(checkpoint['net'])
        model.eval()
        return model
    
    def set_intrinsic_parameter(self, width, height) -> None:
        self.camera.set_camera_matrix(width, height)

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        return self._landmark_estimator.detect_faces(image)

    def estimate_gaze(self, image: np.ndarray, face: Face) -> None:
        MODEL3D.estimate_head_pose(face, self.camera)
        MODEL3D.compute_3d_pose(face)
        MODEL3D.compute_face_eye_centers(face)

        if self._config.mode == GazeEstimationMethod.MPIIGaze.name:
            for key in self.EYE_KEYS:
                eye = getattr(face, key.name.lower())
                self._head_pose_normalizer.normalize(image, eye)

                # import cv2
                # cv2.imshow(key.name.lower(), eye.normalized_image)
                # cv2.waitKey(1)

            self._run_mpiigaze_model(face)
            # self._run_blink_model(face)

        elif self._config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
            self._head_pose_normalizer.normalize(image, face)
            self._run_mpiifacegaze_model(face)

    def _run_blink_model(self, face: Face) -> None:
        #eye.normalized_image = cv2.resize(eye.normalized_image, (24, 24))
        images = []
        for key in self.EYE_KEYS:
            eye = getattr(face , key.name.lower())
            image = eye.normalized_image
            image = cv2.resize(image, (24, 24))
            image = self._blink_transform(image)
            images.append(image)
        images = torch.stack(images)

        device = torch.device(self._config.device)
        with torch.no_grad():
            images = images.to(device)
            predictions = self._blink_model(images)
            predictions = torch.nn.functional.softmax(predictions, dim=1)
            predictions = predictions.cpu().numpy()

        for i, key in enumerate(self.EYE_KEYS):
            eye = getattr(face, key.name.lower())
            eye.opened = predictions[i, 0]
            eye.closed = predictions[i, 1]
            # print(predictions[i])
            # print("%s : %f" % (key.name.lower(), eye.opened * 100))

    def _run_mpiigaze_model(self, face: Face) -> None:
        images = []
        head_poses = []
        for key in self.EYE_KEYS:
            eye = getattr(face, key.name.lower())
            image = eye.normalized_image
            normalized_head_pose = eye.normalized_head_rot2d
            if key == FacePartsName.REYE:
                image = image[:, ::-1]
                normalized_head_pose *= np.array([1, -1])
            image = self._transform(image)
            images.append(image)
            head_poses.append(normalized_head_pose)
        images = torch.stack(images)
        head_poses = np.array(head_poses).astype(np.float32)
        head_poses = torch.from_numpy(head_poses)

        device = torch.device(self._config.device)
        with torch.no_grad():
            images = images.to(device)
            head_poses = head_poses.to(device)
            predictions = self._gaze_estimation_model(images, head_poses)
            predictions = predictions.cpu().numpy()

        for i, key in enumerate(self.EYE_KEYS):
            eye = getattr(face, key.name.lower())
            eye.normalized_gaze_angles = predictions[i]
            if key == FacePartsName.REYE:
                eye.normalized_gaze_angles *= np.array([1, -1])
            eye.angle_to_vector()
            eye.denormalize_gaze_vector()

    def _run_mpiifacegaze_model(self, face: Face) -> None:
        image = self._transform(face.normalized_image).unsqueeze(0)

        device = torch.device(self._config.device)
        with torch.no_grad():
            image = image.to(device)
            prediction = self._gaze_estimation_model(image)
            prediction = prediction.cpu().numpy()

        face.normalized_gaze_angles = prediction[0]
        face.angle_to_vector()
        face.denormalize_gaze_vector()
