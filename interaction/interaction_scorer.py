import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, deque
from utils.logger import get_logger


class InteractionScorer:

    def __init__(self, config: Dict):
        self.logger = get_logger()
        self.config = config

        self.proximity_weight = config.get('proximity', {}).get('weight', 0.3)
        self.pose_weight = config.get('pose', {}).get('weight', 0.25)
        self.motion_weight = config.get('motion', {}).get('weight', 0.2)
        self.hoi_weight = config.get('hoi', {}).get('weight', 0.25)

        total_weight = self.proximity_weight + self.pose_weight + self.motion_weight + self.hoi_weight
        if total_weight > 0:
            self.proximity_weight /= total_weight
            self.pose_weight /= total_weight
            self.motion_weight /= total_weight
            self.hoi_weight /= total_weight

        self.temporal_window = config.get('motion', {}).get('temporal_window', 15)
        self.track_positions = defaultdict(lambda: deque(maxlen=self.temporal_window))
        self.standing_threshold = config.get('idle_detection', {}).get('standing_threshold', 3.0)
        self.walking_speed_max = config.get('idle_detection', {}).get('walking_speed_max', 50.0)

        self.logger.info("InteractionScorer initialized")
        self.logger.info(
            f"  Weights - Proximity: {self.proximity_weight:.2f}, "
            f"Pose: {self.pose_weight:.2f}, "
            f"Motion: {self.motion_weight:.2f}, "
            f"HOI: {self.hoi_weight:.2f}"
        )

    def score_frame(
        self,
        person_detections: List[Dict],
        machine_detections: List[Dict],
        poses: Optional[List[np.ndarray]] = None,
        depth_map: Optional[np.ndarray] = None,
        depth_estimator=None
    ) -> Tuple[float, List[Dict]]:
        if len(person_detections) == 0 or len(machine_detections) == 0:
            return 0.0, []

        interactions = []

        for i, person in enumerate(person_detections):
            for j, machine in enumerate(machine_detections):

                if depth_map is not None and depth_estimator is not None:
                    proximity_score = depth_estimator.compute_proximity_score(
                        depth_map, person['bbox'], machine['bbox']
                    )
                else:
                    proximity_score = self._compute_2d_proximity(person['bbox'], machine['bbox'])

                pose_score = 0.0
                if poses is not None and i < len(poses):
                    pose_score = self._compute_pose_score(poses[i], person['bbox'], machine['bbox'])

                motion_score = 0.0
                if 'track_id' in person:
                    motion_score = self._compute_motion_score(person['track_id'], person['bbox'])

                hoi_score = 0.0

                combined_score = (
                    self.proximity_weight * proximity_score +
                    self.pose_weight * pose_score +
                    self.motion_weight * motion_score +
                    self.hoi_weight * hoi_score
                )

                is_idle = self._is_idle(person, poses[i] if poses and i < len(poses) else None)
                if is_idle:
                    combined_score *= 0.1

                interactions.append({
                    'person_idx': i,
                    'machine_idx': j,
                    'person_bbox': person['bbox'],
                    'machine_bbox': machine['bbox'],
                    'person_track_id': person.get('track_id'),
                    'proximity_score': proximity_score,
                    'pose_score': pose_score,
                    'motion_score': motion_score,
                    'hoi_score': hoi_score,
                    'combined_score': combined_score,
                    'is_idle': is_idle
                })

        overall_score = max([i['combined_score'] for i in interactions]) if interactions else 0.0
        return overall_score, interactions

    def _compute_2d_proximity(self, person_bbox: List[float], machine_bbox: List[float]) -> float:
        person_center = np.array([
            (person_bbox[0] + person_bbox[2]) / 2,
            (person_bbox[1] + person_bbox[3]) / 2
        ])
        machine_center = np.array([
            (machine_bbox[0] + machine_bbox[2]) / 2,
            (machine_bbox[1] + machine_bbox[3]) / 2
        ])

        distance = np.linalg.norm(person_center - machine_center)
        person_size = np.sqrt(
            (person_bbox[2] - person_bbox[0]) ** 2 +
            (person_bbox[3] - person_bbox[1]) ** 2
        )

        normalized_distance = distance / person_size if person_size > 0 else float('inf')
        return max(0, 1 - (normalized_distance / 5.0))

    def _compute_pose_score(
        self, pose: np.ndarray, person_bbox: List[float], machine_bbox: List[float]
    ) -> float:
        left_wrist = pose[9]
        right_wrist = pose[10]
        conf_threshold = 0.3

        machine_center = np.array([
            (machine_bbox[0] + machine_bbox[2]) / 2,
            (machine_bbox[1] + machine_bbox[3]) / 2
        ])
        machine_size = np.sqrt(
            (machine_bbox[2] - machine_bbox[0]) ** 2 +
            (machine_bbox[3] - machine_bbox[1]) ** 2
        )

        score = 0.0
        if left_wrist[2] > conf_threshold:
            left_dist = np.linalg.norm(left_wrist[:2] - machine_center)
            score = max(score, max(0, 1 - (left_dist / machine_size)))
        if right_wrist[2] > conf_threshold:
            right_dist = np.linalg.norm(right_wrist[:2] - machine_center)
            score = max(score, max(0, 1 - (right_dist / machine_size)))

        return score

    def _compute_motion_score(self, track_id: int, bbox: List[float]) -> float:
        center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        self.track_positions[track_id].append(center)

        if len(self.track_positions[track_id]) < 2:
            return 0.5

        positions = np.array(list(self.track_positions[track_id]))
        total_movement = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))

        if total_movement < self.standing_threshold:
            return 0.3
        elif total_movement > self.walking_speed_max:
            return 0.4
        else:
            return 0.8

    def _is_idle(self, person: Dict, pose: Optional[np.ndarray]) -> bool:
        if 'track_id' in person:
            track_id = person['track_id']
            if len(self.track_positions[track_id]) >= 2:
                positions = np.array(list(self.track_positions[track_id]))
                total_movement = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
                if total_movement < self.standing_threshold:
                    return True
                if total_movement > self.walking_speed_max:
                    return True
        return False

    def reset(self):
        self.track_positions.clear()
        self.logger.info("Interaction scorer reset")
