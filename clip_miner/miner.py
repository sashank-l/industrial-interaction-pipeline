import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import deque
from utils.logger import get_logger


class ClipMiner:
    def __init__(self, config: Dict, video_loader):
        self.logger = get_logger()
        self.config = config
        self.video_loader = video_loader
        self.clip_duration = config.get('clip_duration', 10.0)
        self.sliding_window = config.get('sliding_window', 5.0)
        self.interaction_threshold = config.get('interaction_threshold', 0.6)
        self.min_interaction_frames = config.get('min_interaction_frames', 60)
        self.nms_threshold = config.get('nms_threshold', 0.5)
        self.save_metadata = config.get('save_metadata', True)

        fps = video_loader.fps
        self.clip_duration_frames = int(self.clip_duration * fps)
        self.sliding_window_frames = int(self.sliding_window * fps)

        self.frame_scores = deque(maxlen=self.clip_duration_frames)
        self.frame_data = deque(maxlen=self.clip_duration_frames)
        self.candidate_clips = []

        self.logger.info("ClipMiner initialized")
        self.logger.info(f"  Clip duration: {self.clip_duration}s ({self.clip_duration_frames} frames)")
        self.logger.info(f"  Sliding window: {self.sliding_window}s ({self.sliding_window_frames} frames)")
        self.logger.info(f"  Interaction threshold: {self.interaction_threshold}")

    def add_frame(self, frame_idx: int, timestamp: float, interaction_score: float, interaction_details: List[Dict]):
        self.frame_scores.append(interaction_score)
        self.frame_data.append({
            'frame_idx': frame_idx,
            'timestamp': timestamp,
            'score': interaction_score,
            'details': interaction_details
        })

        if len(self.frame_scores) >= self.clip_duration_frames:
            self._evaluate_window()

    def _evaluate_window(self):
        scores_array = np.array(list(self.frame_scores))
        high_score_frames = np.sum(scores_array >= self.interaction_threshold)

        if high_score_frames >= self.min_interaction_frames:
            aggregate_score = np.mean(scores_array)
            start_frame_data = self.frame_data[0]
            end_frame_data = self.frame_data[-1]

            candidate = {
                'start_frame': start_frame_data['frame_idx'],
                'end_frame': end_frame_data['frame_idx'],
                'start_time': start_frame_data['timestamp'],
                'end_time': end_frame_data['timestamp'],
                'aggregate_score': aggregate_score,
                'high_score_frames': high_score_frames,
                'frame_details': list(self.frame_data)
            }

            self.candidate_clips.append(candidate)
            self.logger.debug(
                f"Candidate clip: {start_frame_data['timestamp']:.1f}s - "
                f"{end_frame_data['timestamp']:.1f}s, score: {aggregate_score:.2f}"
            )

    def finalize_clips(self, output_dir: str, visualizer=None) -> List[Dict]:
        self.logger.info(f"Finalizing clips from {len(self.candidate_clips)} candidates")

        if len(self.candidate_clips) == 0:
            self.logger.warning("No candidate clips found")
            return []

        selected_clips = self._apply_nms(self.candidate_clips)
        self.logger.info(f"Selected {len(selected_clips)} clips after NMS")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        extracted_clips = []
        save_vis = self.config.get('save_visualization', False)

        for i, clip in enumerate(selected_clips):
            clip_id = f"clip_{i+1:04d}"
            clip_filename = f"{clip_id}.mp4"
            clip_path = output_path / clip_filename

            if save_vis and visualizer is not None:
                success = self.video_loader.extract_annotated_clip(
                    clip['start_time'], clip['end_time'],
                    str(clip_path), clip['frame_details'], visualizer
                )
            else:
                success = self.video_loader.extract_clip(
                    clip['start_time'], clip['end_time'], str(clip_path)
                )

            if success:
                metadata = {
                    'clip_id': clip_id,
                    'filename': clip_filename,
                    'start_time': clip['start_time'],
                    'end_time': clip['end_time'],
                    'duration': clip['end_time'] - clip['start_time'],
                    'start_frame': clip['start_frame'],
                    'end_frame': clip['end_frame'],
                    'interaction_score': float(clip['aggregate_score']),
                    'high_score_frames': int(clip['high_score_frames']),
                    'total_frames': len(clip['frame_details'])
                }

                if self.save_metadata:
                    metadata_path = output_path / f"{clip_id}.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)

                extracted_clips.append(metadata)
                self.logger.info(
                    f"Extracted {clip_id}: {clip['start_time']:.1f}s - "
                    f"{clip['end_time']:.1f}s (score: {clip['aggregate_score']:.2f})"
                )

        return extracted_clips

    def _apply_nms(self, clips: List[Dict]) -> List[Dict]:
        if len(clips) == 0:
            return []

        sorted_clips = sorted(clips, key=lambda x: x['aggregate_score'], reverse=True)
        selected = []

        for clip in sorted_clips:
            overlaps = False
            for selected_clip in selected:
                iou = self._compute_temporal_iou(clip, selected_clip)
                if iou > self.nms_threshold:
                    overlaps = True
                    break
            if not overlaps:
                selected.append(clip)

        return selected

    @staticmethod
    def _compute_temporal_iou(clip1: Dict, clip2: Dict) -> float:
        start1, end1 = clip1['start_time'], clip1['end_time']
        start2, end2 = clip2['start_time'], clip2['end_time']

        inter_start = max(start1, start2)
        inter_end = min(end1, end2)
        intersection = max(0, inter_end - inter_start)
        union = (end1 - start1) + (end2 - start2) - intersection

        return intersection / union if union > 0 else 0

    def reset(self):
        self.frame_scores.clear()
        self.frame_data.clear()
        self.candidate_clips.clear()
        self.logger.info("Clip miner reset")
