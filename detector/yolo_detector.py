import torch
import numpy as np
from typing import List, Dict, Tuple
from ultralytics import YOLO
from utils.logger import get_logger


class YOLODetector:

    COCO_CLASSES = {
        0: 'person',
        2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck',
        56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
        61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
        66: 'keyboard', 67: 'cell phone', 73: 'book', 74: 'clock', 75: 'vase',
        76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
    }

    def __init__(
        self,
        model_name: str = "yolov8x.pt",
        img_size: int = 1280,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        person_class: int = 0,
        machine_classes: List[int] = None,
        device: str = "cuda"
    ):
        self.logger = get_logger()
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.person_class = person_class

        if machine_classes is None:
            machine_classes = [2, 3, 5, 7, 56, 57, 58, 59, 60, 61, 62, 63]
        self.machine_classes = machine_classes

        if device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"
        self.device = device

        self.logger.info(f"Loading YOLOv8 model: {model_name}")
        self.model = YOLO(model_name)
        self.model.to(device)

        self.logger.info(f"YOLODetector initialized")
        self.logger.info(f"  Device: {device}")
        self.logger.info(f"  Image size: {img_size}")
        self.logger.info(f"  Confidence threshold: {conf_threshold}")

    def detect(self, image: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        results = self.model.predict(
            image,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device
        )

        person_detections = []
        machine_detections = []

        if len(results) > 0:
            result = results[0]
            boxes = result.boxes

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())

                detection = {
                    'bbox': xyxy.tolist(),
                    'conf': conf,
                    'class_id': class_id,
                    'class_name': self.COCO_CLASSES.get(class_id, f'class_{class_id}')
                }

                if class_id == self.person_class:
                    person_detections.append(detection)
                elif class_id in self.machine_classes:
                    machine_detections.append(detection)

        return person_detections, machine_detections

    def detect_batch(self, images: List[np.ndarray]) -> List[Tuple[List[Dict], List[Dict]]]:
        results_list = self.model.predict(
            images,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device
        )

        all_detections = []

        for result in results_list:
            person_detections = []
            machine_detections = []
            boxes = result.boxes

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())

                detection = {
                    'bbox': xyxy.tolist(),
                    'conf': conf,
                    'class_id': class_id,
                    'class_name': self.COCO_CLASSES.get(class_id, f'class_{class_id}')
                }

                if class_id == self.person_class:
                    person_detections.append(detection)
                elif class_id in self.machine_classes:
                    machine_detections.append(detection)

            all_detections.append((person_detections, machine_detections))

        return all_detections

    def get_person_crops(self, image: np.ndarray, person_detections: List[Dict]) -> List[np.ndarray]:
        crops = []
        for det in person_detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            crop = image[y1:y2, x1:x2]
            crops.append(crop)
        return crops
