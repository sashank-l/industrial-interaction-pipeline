import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from utils.logger import get_logger


class STrack:

    _count = 0

    def __init__(self, tlwh, score, class_id):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = score
        self.class_id = class_id
        self.track_id = None
        self.is_activated = False
        self.state = 'new'
        self.frame_id = 0
        self.tracklet_len = 0
        self.start_frame = 0

    def activate(self, frame_id):
        self.track_id = self.next_id()
        self.tracklet_len = 0
        self.state = 'tracked'
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id):
        self.tlwh = new_track.tlwh
        self.score = new_track.score
        self.tracklet_len = 0
        self.state = 'tracked'
        self.is_activated = True
        self.frame_id = frame_id

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.tlwh = new_track.tlwh
        self.score = new_track.score
        self.state = 'tracked'
        self.is_activated = True

    def mark_lost(self):
        self.state = 'lost'

    def mark_removed(self):
        self.state = 'removed'

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def next_id():
        STrack._count += 1
        return STrack._count

    @staticmethod
    def reset_id():
        STrack._count = 0


class ByteTracker:

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        min_box_area: int = 100
    ):
        self.logger = get_logger()
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area

        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0

        self.logger.info("ByteTracker initialized")
        self.logger.info(f"  Track threshold: {track_thresh}")
        self.logger.info(f"  Track buffer: {track_buffer}")
        self.logger.info(f"  Match threshold: {match_thresh}")

    def update(self, detections: List[Dict]) -> List[Dict]:
        self.frame_id += 1

        valid_detections = [
            det for det in detections
            if (det['bbox'][2] - det['bbox'][0]) * (det['bbox'][3] - det['bbox'][1]) >= self.min_box_area
        ]

        if len(valid_detections) == 0:
            for track in self.tracked_stracks:
                track.mark_lost()
            self.lost_stracks.extend(self.tracked_stracks)
            self.tracked_stracks = []
            return []

        detections_high, detections_low = [], []
        for det in valid_detections:
            x1, y1, x2, y2 = det['bbox']
            track = STrack([x1, y1, x2 - x1, y2 - y1], det['conf'], det['class_id'])
            (detections_high if det['conf'] >= self.track_thresh else detections_low).append(track)

        unconfirmed, tracked_stracks = [], []
        for track in self.tracked_stracks:
            (unconfirmed if not track.is_activated else tracked_stracks).append(track)

        matches, u_track, u_detection = self._match(tracked_stracks, detections_high)
        for itracked, idet in matches:
            tracked_stracks[itracked].update(detections_high[idet], self.frame_id)

        for i in u_detection:
            detections_high[i].activate(self.frame_id)
            tracked_stracks.append(detections_high[i])

        if len(detections_low) > 0:
            matches_low, u_track_low, _ = self._match(
                [tracked_stracks[i] for i in u_track], detections_low
            )
            for itracked, idet in matches_low:
                tracked_stracks[u_track[itracked]].update(detections_low[idet], self.frame_id)

        for i in u_track:
            tracked_stracks[i].mark_lost()
            self.lost_stracks.append(tracked_stracks[i])

        self.tracked_stracks = [t for t in tracked_stracks if t.state == 'tracked']
        self.lost_stracks = [
            t for t in self.lost_stracks if self.frame_id - t.frame_id <= self.track_buffer
        ]

        return [
            {'bbox': list(track.tlbr), 'conf': track.score, 'class_id': track.class_id, 'track_id': track.track_id}
            for track in self.tracked_stracks
        ]

    def _match(self, tracks, detections):
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._iou(track.tlbr, det.tlbr)

        matches = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_detections = list(range(len(detections)))

        while iou_matrix.size > 0 and iou_matrix.max() > self.match_thresh:
            i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            matches.append((i, j))
            if i in unmatched_tracks:
                unmatched_tracks.remove(i)
            if j in unmatched_detections:
                unmatched_detections.remove(j)
            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0

        return matches, unmatched_tracks, unmatched_detections

    @staticmethod
    def _iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def reset(self):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        STrack.reset_id()
        self.logger.info("Tracker reset")
