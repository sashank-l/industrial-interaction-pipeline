import cv2
import torch
import numpy as np
from typing import Tuple, Optional
from utils.logger import get_logger


class MiDaSDepth:

    def __init__(
        self,
        model_type: str = "DPT_Large",
        optimize: bool = True,
        height: int = 384,
        square: bool = False,
        device: str = "cuda"
    ):
        self.logger = get_logger()
        self.model_type = model_type
        self.height = height
        self.square = square

        if device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA not available for depth, falling back to CPU")
            device = "cpu"
        self.device = device

        try:
            self.logger.info(f"Loading MiDaS model: {model_type}")
            self.model = torch.hub.load("intel-isl/MiDaS", model_type)
            self.model.to(device)
            self.model.eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if model_type in ("DPT_Large", "DPT_Hybrid"):
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform

            self.logger.info(f"MiDaS depth estimator initialized")
            self.logger.info(f"  Device: {device}")
            self.logger.info(f"  Model: {model_type}")

        except Exception as e:
            self.logger.error(f"Failed to load MiDaS: {e}")
            self.logger.warning("Depth estimation will be disabled")
            self.model = None
            self.transform = None

    def estimate(self, image: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None:
            return None

        try:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_batch = self.transform(img_rgb).to(self.device)

            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            return prediction.cpu().numpy()

        except Exception as e:
            self.logger.error(f"Depth estimation failed: {e}")
            return None

    def get_depth_at_bbox(self, depth_map: np.ndarray, bbox: Tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = depth_map.shape
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return 0.0
        return float(np.mean(depth_map[y1:y2, x1:x2]))

    def compute_distance(
        self,
        depth_map: np.ndarray,
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float]
    ) -> float:
        depth1 = self.get_depth_at_bbox(depth_map, bbox1)
        depth2 = self.get_depth_at_bbox(depth_map, bbox2)
        distance = abs(depth1 - depth2)
        max_depth = np.max(depth_map)
        if max_depth > 0:
            distance = distance / max_depth
        return distance

    def compute_proximity_score(
        self,
        depth_map: np.ndarray,
        person_bbox: Tuple[float, float, float, float],
        machine_bbox: Tuple[float, float, float, float],
        max_distance: float = 2.0
    ) -> float:
        depth_distance = self.compute_distance(depth_map, person_bbox, machine_bbox)

        person_center = np.array([
            (person_bbox[0] + person_bbox[2]) / 2,
            (person_bbox[1] + person_bbox[3]) / 2
        ])
        machine_center = np.array([
            (machine_bbox[0] + machine_bbox[2]) / 2,
            (machine_bbox[1] + machine_bbox[3]) / 2
        ])

        h, w = depth_map.shape
        spatial_distance_norm = np.linalg.norm(person_center - machine_center) / np.sqrt(h**2 + w**2)
        combined_distance = (depth_distance + spatial_distance_norm) / 2

        return max(0, 1 - (combined_distance / max_distance))
