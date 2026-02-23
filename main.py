import sys
import yaml
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import setup_logger, get_logger
from utils.visualization import Visualizer
from video_loader import VideoLoader
from detector import YOLODetector
from tracker import ByteTracker
from pose import ViTPoseEstimator
from depth import MiDaSDepth
from interaction import InteractionScorer
from clip_miner import ClipMiner


class IndustrialInteractionPipeline:

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        log_config = self.config.get('logging', {})
        self.logger = setup_logger(
            level=log_config.get('level', 'INFO'),
            log_file=log_config.get('log_file') if log_config.get('save_logs') else None
        )

        self.logger.info("=" * 80)
        self.logger.info("Industrial Interaction Clip Mining Pipeline")
        self.logger.info("=" * 80)

        self._check_gpu()
        self._initialize_modules()
        self.visualizer = Visualizer()

    def _check_gpu(self):
        device_config = self.config.get('device', {})
        use_gpu = device_config.get('use_gpu', True)

        if use_gpu and torch.cuda.is_available():
            gpu_id = device_config.get('gpu_id', 0)
            gpu_name = torch.cuda.get_device_name(gpu_id)
            self.device = f"cuda:{gpu_id}"
            self.logger.info(f"✓ GPU detected: {gpu_name}")
            self.logger.info(f"  CUDA version: {torch.version.cuda}")
        else:
            self.device = "cpu"
            if use_gpu:
                self.logger.warning("⚠ GPU requested but not available, using CPU")
            else:
                self.logger.info("Using CPU")

    def _initialize_modules(self):
        self.logger.info("\nInitializing modules...")

        det_config = self.config.get('detection', {})
        self.detector = YOLODetector(
            model_name=det_config.get('model', 'yolov8x.pt'),
            img_size=det_config.get('img_size', 1280),
            conf_threshold=det_config.get('conf_threshold', 0.25),
            iou_threshold=det_config.get('iou_threshold', 0.45),
            person_class=det_config.get('person_class', 0),
            machine_classes=det_config.get('machine_classes'),
            device=self.device
        )

        track_config = self.config.get('tracking', {})
        self.tracker_person = ByteTracker(
            track_thresh=track_config.get('track_thresh', 0.5),
            track_buffer=track_config.get('track_buffer', 30),
            match_thresh=track_config.get('match_thresh', 0.8),
            min_box_area=track_config.get('min_box_area', 100)
        )
        self.tracker_machine = ByteTracker(
            track_thresh=track_config.get('track_thresh', 0.5),
            track_buffer=track_config.get('track_buffer', 30),
            match_thresh=track_config.get('match_thresh', 0.8),
            min_box_area=track_config.get('min_box_area', 100)
        )

        pose_config = self.config.get('pose', {})
        self.pose_estimator = ViTPoseEstimator(
            model_name=pose_config.get('model', 'vitpose-b'),
            conf_threshold=pose_config.get('conf_threshold', 0.3),
            device=self.device
        )

        depth_config = self.config.get('depth', {})
        self.depth_estimator = MiDaSDepth(
            model_type=depth_config.get('model', 'DPT_Large'),
            optimize=depth_config.get('optimize', True),
            height=depth_config.get('height', 384),
            square=depth_config.get('square', False),
            device=self.device
        )

        interaction_config = self.config.get('interaction', {})
        self.interaction_scorer = InteractionScorer(interaction_config)

        self.logger.info("✓ All modules initialized\n")

    def process_video(self, video_path: str, output_dir: str = None):
        if output_dir is None:
            output_dir = self.config.get('video', {}).get('output_dir', 'output')

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Processing video: {video_path}")
        self.logger.info(f"Output directory: {output_dir}\n")

        frame_skip = self.config.get('performance', {}).get('frame_skip', 1)
        video_loader = VideoLoader(video_path, frame_skip=frame_skip)

        clip_config = self.config.get('clip_mining', {})
        clip_miner = ClipMiner(clip_config, video_loader)

        max_frames = self.config.get('performance', {}).get('max_frames')
        show_progress = self.config.get('logging', {}).get('progress_bar', True)

        if max_frames:
            expected_frames = min(max_frames, video_loader.total_frames // frame_skip)
        else:
            expected_frames = video_loader.total_frames // frame_skip

        self.logger.info("Starting frame processing...")
        self.logger.info(f"Expected to process: {expected_frames} frames")

        processed_frames = 0
        pbar = tqdm(total=expected_frames, desc="Processing frames") if show_progress else None

        for frame_idx, timestamp, frame in video_loader:
            if max_frames and frame_idx >= max_frames:
                break

            person_dets, machine_dets = self.detector.detect(frame)
            person_dets_tracked = self.tracker_person.update(person_dets)
            machine_dets_tracked = self.tracker_machine.update(machine_dets)

            poses = None
            if len(person_dets_tracked) > 0:
                poses = self.pose_estimator.estimate(frame, person_dets_tracked)

            depth_map = self.depth_estimator.estimate(frame)

            interaction_score, interaction_details = self.interaction_scorer.score_frame(
                person_dets_tracked,
                machine_dets_tracked,
                poses=poses,
                depth_map=depth_map,
                depth_estimator=self.depth_estimator
            )

            all_dets = []
            for d in person_dets_tracked:
                all_dets.append({'bbox': d['bbox'], 'class': 'person', 'track_id': d.get('track_id')})
            for d in machine_dets_tracked:
                all_dets.append({'bbox': d['bbox'], 'class': 'machine', 'track_id': d.get('track_id')})

            rich_details = {
                'interactions': interaction_details,
                'detections': all_dets,
                'poses': poses,
            }
            clip_miner.add_frame(frame_idx, timestamp, interaction_score, rich_details)

            processed_frames += 1
            if pbar:
                pbar.update(1)

            if processed_frames % 100 == 0:
                self.logger.debug(f"Processed {processed_frames} frames, current score: {interaction_score:.2f}")

        if pbar:
            pbar.close()

        self.logger.info(f"\nProcessed {processed_frames} frames")
        self.logger.info("\nExtracting clips...")
        extracted_clips = clip_miner.finalize_clips(output_dir, visualizer=self.visualizer)

        self._save_summary(extracted_clips, output_path, video_loader)
        video_loader.close()

        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"Pipeline completed! Extracted {len(extracted_clips)} clips")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info("=" * 80)

        return extracted_clips

    def _save_summary(self, clips: list, output_path: Path, video_loader):
        import json
        summary = {
            'video_info': video_loader.get_metadata(),
            'total_clips': len(clips),
            'clips': clips,
            'config': self.config
        }
        summary_path = output_path / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        self.logger.info(f"Summary saved to: {summary_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Industrial Interaction Clip Mining Pipeline")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--max_frames', type=int, default=None)

    args = parser.parse_args()

    pipeline = IndustrialInteractionPipeline(args.config)

    if args.max_frames is not None:
        if 'performance' not in pipeline.config:
            pipeline.config['performance'] = {}
        pipeline.config['performance']['max_frames'] = args.max_frames

    pipeline.process_video(args.input, args.output)


if __name__ == '__main__':
    main()
