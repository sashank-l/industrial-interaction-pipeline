"""
Visualization utilities for debugging and output.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict


class Visualizer:
    """Visualization utilities for the pipeline."""
    
    # COCO pose skeleton connections
    SKELETON = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
        [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
        [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
        [2, 4], [3, 5], [4, 6], [5, 7]
    ]
    
    COLORS = {
        'person': (0, 255, 0),      # Green
        'machine': (255, 0, 0),     # Blue
        'interaction': (0, 0, 255), # Red
        'skeleton': (255, 255, 0),  # Cyan
        'keypoint': (0, 255, 255),  # Yellow
    }
    
    def __init__(self):
        pass
    
    def draw_bbox(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        label: str = "",
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        track_id: Optional[int] = None
    ) -> np.ndarray:
        """
        Draw bounding box on image.
        
        Args:
            image: Input image
            bbox: Bounding box (x1, y1, x2, y2)
            label: Label text
            color: Box color (BGR)
            thickness: Line thickness
            track_id: Optional track ID
        
        Returns:
            Image with bounding box
        """
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        if label or track_id is not None:
            text = f"{label}"
            if track_id is not None:
                text += f" ID:{track_id}"
            
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(image, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return image
    
    def draw_pose(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        conf_threshold: float = 0.3
    ) -> np.ndarray:
        """
        Draw pose skeleton on image.
        
        Args:
            image: Input image
            keypoints: Keypoints array (17, 3) - [x, y, conf]
            conf_threshold: Minimum confidence to draw
        
        Returns:
            Image with pose skeleton
        """
        # Draw skeleton connections
        for connection in self.SKELETON:
            pt1_idx, pt2_idx = connection[0] - 1, connection[1] - 1
            
            if pt1_idx >= len(keypoints) or pt2_idx >= len(keypoints):
                continue
            
            pt1 = keypoints[pt1_idx]
            pt2 = keypoints[pt2_idx]
            
            if pt1[2] > conf_threshold and pt2[2] > conf_threshold:
                pt1_pos = (int(pt1[0]), int(pt1[1]))
                pt2_pos = (int(pt2[0]), int(pt2[1]))
                cv2.line(image, pt1_pos, pt2_pos, self.COLORS['skeleton'], 2)
        
        # Draw keypoints
        for kp in keypoints:
            if kp[2] > conf_threshold:
                cv2.circle(image, (int(kp[0]), int(kp[1])), 4, self.COLORS['keypoint'], -1)
        
        return image
    
    def draw_depth_map(
        self,
        depth: np.ndarray,
        colormap: int = cv2.COLORMAP_INFERNO
    ) -> np.ndarray:
        """
        Convert depth map to colored visualization.
        
        Args:
            depth: Depth map (H, W)
            colormap: OpenCV colormap
        
        Returns:
            Colored depth map (H, W, 3)
        """
        # Normalize to 0-255
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_norm, colormap)
        return depth_colored
    
    def draw_interaction_score(
        self,
        image: np.ndarray,
        score: float,
        position: Tuple[int, int] = (10, 30)
    ) -> np.ndarray:
        """
        Draw interaction score on image.
        
        Args:
            image: Input image
            score: Interaction score (0-1)
            position: Text position
        
        Returns:
            Image with score overlay
        """
        # Color based on score
        if score > 0.7:
            color = (0, 255, 0)  # Green - high interaction
        elif score > 0.4:
            color = (0, 255, 255)  # Yellow - medium
        else:
            color = (0, 0, 255)  # Red - low
        
        text = f"Interaction: {score:.2f}"
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        return image
    
    def create_overlay(
        self,
        image: np.ndarray,
        detections: List[Dict],
        poses: Optional[List[np.ndarray]] = None,
        depth: Optional[np.ndarray] = None,
        interaction_score: Optional[float] = None
    ) -> np.ndarray:
        """
        Create complete visualization overlay.
        
        Args:
            image: Input image
            detections: List of detection dicts with 'bbox', 'class', 'track_id'
            poses: List of pose keypoints
            depth: Depth map
            interaction_score: Overall interaction score
        
        Returns:
            Annotated image
        """
        vis_image = image.copy()
        
        # Draw depth map (side by side or overlay)
        if depth is not None:
            depth_vis = self.draw_depth_map(depth)
            depth_vis = cv2.resize(depth_vis, (image.shape[1], image.shape[0]))
            vis_image = cv2.addWeighted(vis_image, 0.7, depth_vis, 0.3, 0)
        
        # Draw detections
        for det in detections:
            bbox = det['bbox']
            class_name = det.get('class', 'unknown')
            track_id = det.get('track_id')
            
            color = self.COLORS['person'] if class_name == 'person' else self.COLORS['machine']
            self.draw_bbox(vis_image, bbox, class_name, color, track_id=track_id)
        
        # Draw poses
        if poses is not None:
            for pose in poses:
                self.draw_pose(vis_image, pose)
        
        # Draw interaction score
        if interaction_score is not None:
            self.draw_interaction_score(vis_image, interaction_score)
        
        return vis_image
