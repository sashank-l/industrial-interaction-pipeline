import cv2
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from utils.logger import get_logger


class ViTPoseEstimator:

    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    def __init__(self, model_name: str = "vitpose-b", conf_threshold: float = 0.3, device: str = "cuda"):
        self.logger = get_logger()
        self.conf_threshold = conf_threshold

        if device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA not available for pose, falling back to CPU")
            device = "cpu"
        self.device = device

        self.use_mmpose = False
        try:
            from mmpose.apis import init_model, inference_topdown
            from mmpose.structures import merge_data_samples
            self.logger.warning("MMPose integration requires model configuration")
            self.logger.warning("Using simplified pose estimation for now")
        except ImportError:
            self.logger.warning("MMPose not available, using simplified pose estimation")

        self.logger.info(f"ViTPoseEstimator initialized")
        self.logger.info(f"  Device: {device}")
        self.logger.info(f"  Using MMPose: {self.use_mmpose}")

    def estimate(self, image: np.ndarray, person_detections: List[Dict]) -> List[np.ndarray]:
        if len(person_detections) == 0:
            return []

        poses = []
        for det in person_detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

            person_crop = image[y1:y2, x1:x2]
            if person_crop.size == 0:
                poses.append(np.zeros((17, 3)))
                continue

            pose = self._estimate_simple_pose(person_crop, (x1, y1))
            poses.append(pose)

        return poses

    def _estimate_simple_pose(self, person_crop: np.ndarray, offset: Tuple[int, int]) -> np.ndarray:
        h, w = person_crop.shape[:2]
        x_off, y_off = offset
        keypoints = np.zeros((17, 3))

        keypoints[0]  = [x_off + w * 0.5,  y_off + h * 0.15, 0.7]   # nose
        keypoints[1]  = [x_off + w * 0.45, y_off + h * 0.12, 0.6]   # left eye
        keypoints[2]  = [x_off + w * 0.55, y_off + h * 0.12, 0.6]   # right eye
        keypoints[3]  = [x_off + w * 0.40, y_off + h * 0.15, 0.5]   # left ear
        keypoints[4]  = [x_off + w * 0.60, y_off + h * 0.15, 0.5]   # right ear
        keypoints[5]  = [x_off + w * 0.35, y_off + h * 0.30, 0.8]   # left shoulder
        keypoints[6]  = [x_off + w * 0.65, y_off + h * 0.30, 0.8]   # right shoulder
        keypoints[7]  = [x_off + w * 0.30, y_off + h * 0.50, 0.7]   # left elbow
        keypoints[8]  = [x_off + w * 0.70, y_off + h * 0.50, 0.7]   # right elbow
        keypoints[9]  = [x_off + w * 0.25, y_off + h * 0.65, 0.7]   # left wrist
        keypoints[10] = [x_off + w * 0.75, y_off + h * 0.65, 0.7]   # right wrist
        keypoints[11] = [x_off + w * 0.40, y_off + h * 0.70, 0.6]   # left hip
        keypoints[12] = [x_off + w * 0.60, y_off + h * 0.70, 0.6]   # right hip
        keypoints[13] = [x_off + w * 0.38, y_off + h * 0.85, 0.5]   # left knee
        keypoints[14] = [x_off + w * 0.62, y_off + h * 0.85, 0.5]   # right knee
        keypoints[15] = [x_off + w * 0.36, y_off + h * 0.98, 0.4]   # left ankle
        keypoints[16] = [x_off + w * 0.64, y_off + h * 0.98, 0.4]   # right ankle

        return keypoints

    def get_hand_positions(self, pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return pose[9], pose[10]

    def get_arm_keypoints(self, pose: np.ndarray) -> np.ndarray:
        return pose[[5, 6, 7, 8, 9, 10]]

    def is_reaching(self, pose: np.ndarray, threshold: float = 0.3) -> bool:
        left_shoulder  = pose[5]
        right_shoulder = pose[6]
        left_wrist     = pose[9]
        right_wrist    = pose[10]

        if left_shoulder[2] < self.conf_threshold or left_wrist[2] < self.conf_threshold:
            return False
        if right_shoulder[2] < self.conf_threshold or right_wrist[2] < self.conf_threshold:
            return False

        shoulder_width  = np.linalg.norm(right_shoulder[:2] - left_shoulder[:2])
        left_extension  = np.linalg.norm(left_wrist[:2] - left_shoulder[:2])
        right_extension = np.linalg.norm(right_wrist[:2] - right_shoulder[:2])

        if shoulder_width > 0:
            return (left_extension / shoulder_width) > threshold or (right_extension / shoulder_width) > threshold

        return False
