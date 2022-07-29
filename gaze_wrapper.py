import time

import argparse
import logging
import cv2
import numpy as np
import yaml
from fvcore.common.config import CfgNode

from gaze_estimation.gaze_estimator.common import (Face, FacePartsName, Visualizer)
from gaze_estimation.utils import load_config
from gaze_estimation import GazeEstimationMethod, GazeEstimator

from TDDFA import TDDFA
from utils.pose import viz_pose, calc_pose
from utils.functions import draw_landmarks, get_suffix
from utils.tddfa_util import str2bool

class EvalResult:
    def __init__(self, num_faces: int=None):
        self.num_faces = num_faces
        self.face_pitch: int = None
        self.face_yaw: int = None
        self.face_distance: int = None
        self.eye_opened: int = None
        self.eye_pitch: int = None
        self.eye_yaw: int = None
        self.screen_x: float = None
        self.screen_y: float = None
        self.gaze_direction: int = None

class GazeTracker:
    #def __init__(self, logging):
    def __init__(self):
        #self.logger = logging.getLogger(__name__)
        self.gaze_cfg = load_config()
        self.gaze_estimator = GazeEstimator(self.gaze_cfg)
        self.visualizer = Visualizer(self.gaze_estimator.camera)
        self.DETECTOR_TYPE = "3DDFA"

        self.tddfa_args = CfgNode()
        self.tddfa_args.config = 'configs/mb1_120x120.yml'
        self.tddfa_args.mode = 'cpu'
        self.tddfa_args.opt = '2d_sparse'  # '2d_sparse', '2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'pose', 'ply', 'obj'
        self.tddfa_args.show_flag = False
        self.tddfa_args.onnx = True

        self.tddfa_cfg = yaml.load(open(self.tddfa_args.config), Loader=yaml.SafeLoader)
        self.dense_flag = self.tddfa_args.opt in ('2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'ply', 'obj')

        # show for debug
        self.show_bbox = self.gaze_cfg.demo.show_bbox
        self.show_head_pose = self.gaze_cfg.demo.show_head_pose
        self.show_landmarks = self.gaze_cfg.demo.show_landmarks
        self.show_normalized_image = self.gaze_cfg.demo.show_normalized_image
        self.show_template_model = self.gaze_cfg.demo.show_template_model

        self._eyes = None # eye images

    @property
    def eyes(self):
        return self._eyes

    def __call__(self, frame):
        self.visualizer.set_image(frame.copy())

        eval_result = EvalResult()
        # Init FaceBoxes and TDDFA, recommend using onnx flag
        if self.tddfa_args.onnx:
            import os
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            os.environ['OMP_NUM_THREADS'] = '2'

            from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
            from TDDFA_ONNX import TDDFA_ONNX

            face_boxes = FaceBoxes_ONNX()
            tddfa = TDDFA_ONNX(**self.tddfa_cfg)

        else:            
            gpu_mode = self.tddfa_args.mode == 'gpu'
            tddfa = TDDFA(gpu_mode=gpu_mode, **self.tddfa_cfg)
            face_boxes = FaceBoxes()
        print('++++')
           

        # Detect faces, get 3DMM params and roi boxes
        boxes = face_boxes(frame)
        if len(boxes) == 0:
            return EvalResult()

        param_lst, roi_box_lst = tddfa(frame, boxes)
        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=self.dense_flag)

        print('xxxx')
        if self.tddfa_args.show_flag:
            old_suffix = get_suffix(self.tddfa_args.img_fp)
            new_suffix = f'.{self.tddfa_args.opt}' if self.tddfa_args.opt in ('ply', 'obj') else '.jpg'
            wfp = f'examples/results/{self.tddfa_args.img_fp.split("/")[-1].replace(old_suffix, "")}_landmarks' + new_suffix
            draw_landmarks(frame, ver_lst, show_flag=self.tddfa_args.show_flag, dense_flag=self.dense_flag, wfp=wfp)

            wfp = f'examples/results/{self.tddfa_args.img_fp.split("/")[-1].replace(old_suffix, "")}_pose' + new_suffix
            viz_pose(frame, param_lst, ver_lst, show_flag=self.tddfa_args.show_flag, wfp=wfp)

        # Convert to Face type
        if not type(ver_lst) in [tuple, list]:
            ver_lst = [ver_lst]

        faces = []
        for box, ver, param in zip(roi_box_lst, ver_lst, param_lst):
            np_bbox = np.array([[box[0], box[1]], [box[2], box[3]]])
            np_lms = np.array([(x, y) for x, y in zip(ver[0], ver[1])])

            lms_z = np.array([z for z in ver[2]])
            face_distance = np.mean(lms_z)

            # pose
            P, pose = calc_pose(param)
            # print(P[:, :3])
            # self.logger.info(f'[face] yaw: {pose[0]:.1f}, pitch: {pose[1]:.1f}, roll: {pose[2]:.1f}')

            face = Face(np_bbox, np_lms)
            faces.append(face)
            eval_result.face_pitch = int(pose[1])
            eval_result.face_yaw = int(pose[0])  
            eval_result.face_distance = int(frame.shape[0]/2 - face_distance)
            break

        # Compute gaze angles
        face = faces[0]

        #! set intrinsic param. 
        self.gaze_estimator.set_intrinsic_parameter(frame.shape[1], frame.shape[0])

        self.gaze_estimator.estimate_gaze(frame, face)
        #self._draw_landmarks(face)  #! landmark points 
        # self._draw_gaze_vector(face)
        self._display_normalized_image(face)      
        pt0, pt1 = self._draw_gaze_one_vector(face)
        # self._draw_eye_blink(face)

        intersect_point = self._draw_target_on_screen(face)
        #face_distance = int(frame.shape[0]/2 - face_distance)

        _, single_gaze_vector = self._get_single_gaze_vector(face)

        #! pack response 
        single_eye_pitch, single_eye_yaw = np.rad2deg(self.vector_to_angle(single_gaze_vector))
        eval_result.eye_pitch, eval_result.eye_yaw = int(single_eye_pitch), int(single_eye_yaw)
        eval_result.screen_x, eval_result.screen_y = int(intersect_point[0]*frame.shape[1]), int(intersect_point[1]*frame.shape[0])

        eval_result.gaze_direction = pt0[0] - pt1[0]
        return eval_result
        #return intersect_point, face.leye.opened, face.reye.opened, face_distance

    def eye_aspect_ratio(self, eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])

        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        # return the eye aspect ratio
        return ear

    def get_visualizer_image(self):
        return self.visualizer.image

    def vector_to_angle(self, vector: np.ndarray) -> np.ndarray:
        assert vector.shape == (3, )
        x, y, z = vector
        pitch = np.arcsin(-y)
        yaw = np.arctan2(-x, -z)
        return np.array([pitch, yaw])

    def _get_single_gaze_vector(self, face: Face):
        length = self.gaze_cfg.demo.gaze_visualization_length
        two_eyes_center = np.array([0.0, 0.0, 0.0])
        two_eyes_vector = np.array([0.0, 0.0, 0.0])
        for key in [FacePartsName.REYE, FacePartsName.LEYE]:
            eye = getattr(face, key.name.lower())
            two_eyes_center += eye.center
            two_eyes_vector += eye.gaze_vector
        two_eyes_center /= 2
        two_eyes_vector /= 2
        return two_eyes_center, two_eyes_vector

    def _draw_landmarks(self, face: Face) -> None:
        if not self.show_landmarks:
            return
        self.visualizer.draw_points(face.landmarks,
                                    color=(0, 255, 255),
                                    size=1)

    def _draw_face_template_model(self, face: Face) -> None:
        if not self.show_template_model:
            return
        self.visualizer.draw_3d_points(face.model3d,
                                       color=(255, 0, 525),
                                       size=1)

    def _display_normalized_image(self, face: Face) -> None:
        reye = face.reye.normalized_image
        leye = face.leye.normalized_image

        normalized = np.hstack([reye, leye])
        #normalized = np.hstack([leye, reye]) #! swap

        normalized = normalized[:, ::-1]
        self.visualizer.draw_norm_img(normalized)

        #! cropped eye image
        self._eyes = normalized.copy()

    def _draw_gaze_vector(self, face: Face) -> None:
        length = self.gaze_cfg.demo.gaze_visualization_length
        for key in [FacePartsName.REYE, FacePartsName.LEYE]:
            eye = getattr(face, key.name.lower())
            self.visualizer.draw_3d_line(
                eye.center, eye.center + length * eye.gaze_vector)
            pitch, yaw = np.rad2deg(eye.vector_to_angle(eye.gaze_vector))

    def _draw_gaze_one_vector(self, face: Face) -> None:
        length = self.gaze_cfg.demo.gaze_visualization_length
        two_eyes_center = np.array([0.0, 0.0, 0.0])
        two_eyes_vector = np.array([0.0, 0.0, 0.0])
        for key in [FacePartsName.REYE, FacePartsName.LEYE]:
            eye = getattr(face, key.name.lower())
            two_eyes_center += eye.center
            two_eyes_vector += eye.gaze_vector
        two_eyes_center /= 2
        two_eyes_vector /= 2
        pt0, pt1 = self.visualizer.draw_3d_line(two_eyes_center, two_eyes_center + length * two_eyes_vector)
        return pt0, pt1

    def _draw_target_on_screen(self, face: Face):
        two_eyes_center = np.array([0.0, 0.0, 0.0])
        two_eyes_vector = np.array([0.0, 0.0, 0.0])
        for key in [FacePartsName.REYE, FacePartsName.LEYE]:
            eye = getattr(face, key.name.lower())
            two_eyes_center += eye.center
            two_eyes_vector += eye.gaze_vector
        two_eyes_center /= 2
        two_eyes_vector /= 2
        
        plane_center = np.array([0.0, 0.0, 0.0])
        plane_normal = np.array([0.0, 0.0, 1.0])

        v_result = plane_center - two_eyes_center
        f_result = np.dot(v_result, plane_normal)
        ray_size = f_result / np.dot(two_eyes_vector, plane_normal)
        intersect_point = two_eyes_center + (two_eyes_vector * ray_size)
        return intersect_point

    def _draw_eye_blink(self, face: Face):
        self.visualizer.draw_eye_blink(face)
if __name__ == '__main__':
    # webcam를 실행 
    
    webcam = cv2.VideoCapture(0)
    
    monitor_system = GazeTracker()
    
    while True:
        _ , frame_bgr = webcam.read()
        
        result = monitor_system(frame_bgr)
        
        print(result.face_distance)
        
        cv2.imshow('window', frame_bgr)
        key = cv2.waitKey(33)
        if key == ord('q'):
            break