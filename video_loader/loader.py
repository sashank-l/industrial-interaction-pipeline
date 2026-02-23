import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, Optional, Dict, List
from utils.logger import get_logger


class VideoLoader:
    def __init__(self, video_path: str, frame_skip: int = 1):
        self.logger = get_logger()
        self.video_path = Path(video_path)
        self.frame_skip = frame_skip

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

        self.logger.info(f"Loaded video: {self.video_path.name}")
        self.logger.info(f"  Resolution: {self.width}x{self.height}")
        self.logger.info(f"  FPS: {self.fps:.2f}")
        self.logger.info(f"  Total frames: {self.total_frames}")
        self.logger.info(f"  Duration: {self.duration/3600:.2f} hours")

        self.current_frame_idx = 0

    def get_metadata(self) -> Dict:
        return {
            'path': str(self.video_path),
            'fps': self.fps,
            'total_frames': self.total_frames,
            'width': self.width,
            'height': self.height,
            'duration': self.duration
        }

    def __iter__(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_idx % self.frame_skip == 0:
                timestamp = frame_idx / self.fps if self.fps > 0 else 0
                yield frame_idx, timestamp, frame
            frame_idx += 1
            self.current_frame_idx = frame_idx

    def get_frame_at(self, frame_idx: int) -> Optional[np.ndarray]:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        return frame if ret else None

    def get_frame_at_time(self, timestamp: float) -> Optional[np.ndarray]:
        return self.get_frame_at(int(timestamp * self.fps))

    def extract_clip(self, start_time: float, end_time: float, output_path: str, fps: Optional[float] = None) -> bool:
        try:
            output_fps = fps or self.fps
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, output_fps, (self.width, self.height))

            start_frame = int(start_time * self.fps)
            end_frame = int(end_time * self.fps)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for _ in range(start_frame, end_frame):
                ret, frame = self.cap.read()
                if not ret:
                    break
                out.write(frame)

            out.release()
            self.logger.info(f"Extracted clip: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to extract clip: {e}")
            return False

    def extract_annotated_clip(
        self,
        start_time: float,
        end_time: float,
        output_path: str,
        frame_data: List[Dict],
        visualizer,
        fps: Optional[float] = None
    ) -> bool:
        try:
            output_fps = fps or self.fps
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, output_fps, (self.width, self.height))

            start_frame_idx = int(start_time * self.fps)
            end_frame_idx = int(end_time * self.fps)
            metadata_map = {d['frame_idx']: d for d in frame_data}

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)

            for frame_idx in range(start_frame_idx, end_frame_idx):
                ret, frame = self.cap.read()
                if not ret:
                    break

                if frame_idx in metadata_map:
                    data = metadata_map[frame_idx]
                    details = data.get('details', {})
                    frame = visualizer.create_overlay(
                        frame,
                        detections=details.get('detections', []),
                        poses=details.get('poses'),
                        depth=details.get('depth'),
                        interaction_score=data.get('score')
                    )

                out.write(frame)

            out.release()
            self.logger.info(f"Extracted annotated clip: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to extract annotated clip: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def reset(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame_idx = 0

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.logger.info("Video loader closed")

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
