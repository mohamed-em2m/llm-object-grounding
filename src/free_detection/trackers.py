"""
Tracker manager supporting ByteTrack, SiamONNXTracker, MOSSE, KCF, CSRT, and VitTracker algorithms.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from free_detection.bytetrack import ByteTracker
from free_detection.siamonnx import SiamONNXTracker


class OpenCVTrackerInstance:
    """Manages an OpenCV tracker instance for a single target box."""

    def __init__(self, algo_name: str, frame: np.ndarray, bbox: Tuple[float, float, float, float], label: str, track_id: int):
        self.algo_name = algo_name
        self.label = label
        self.track_id = track_id
        self.tracker = self._create_tracker(algo_name)
        self.is_valid = False

        if self.tracker is not None and frame is not None and frame.size > 0:
            ymin, xmin, ymax, xmax = bbox
            w = max(1.0, xmax - xmin)
            h = max(1.0, ymax - ymin)
            # OpenCV ROI format: (x, y, w, h)
            roi = (float(xmin), float(ymin), float(w), float(h))
            try:
                self.tracker.init(frame, roi)
                self.is_valid = True
            except Exception:
                self.is_valid = False

    def _create_tracker(self, algo_name: str):
        algo = algo_name.upper()
        if algo == "MOSSE" and hasattr(cv2, "TrackerMOSSE_create"):
            return cv2.TrackerMOSSE_create()
        elif algo == "KCF" and hasattr(cv2, "TrackerKCF_create"):
            return cv2.TrackerKCF_create()
        elif algo == "CSRT" and hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        elif algo in ("VIT", "VITTRACKER", "TRACKERVIT") and hasattr(cv2, "TrackerVit_create"):
            try:
                params = cv2.TrackerVit_Params()
                return cv2.TrackerVit_create(params)
            except Exception:
                return cv2.TrackerVit_create()
        return None

    def update(self, frame: np.ndarray) -> Optional[List[Any]]:
        if not self.is_valid or self.tracker is None:
            return None
        try:
            success, box = self.tracker.update(frame)
            if success:
                x, y, w, h = box
                ymin, xmin = float(y), float(x)
                ymax, xmax = float(y + h), float(x + w)
                return [ymin, xmin, ymax, xmax, self.label, self.track_id]
        except Exception:
            self.is_valid = False
        return None


class MultiAlgorithmTracker:
    """
    Unified object tracking coordinator supporting:
    - ByteTrack (Kalman + Hungarian IoU)
    - SiamONNX (Template & Search ONNX sessions)
    - MOSSE (Ultra-fast ~450+ FPS)
    - KCF (Fast ~80-120 FPS)
    - CSRT (Accurate ~25-40 FPS)
    - VitTracker (Vision Transformer real-time)
    """

    SUPPORTED_ALGOS = [
        "ByteTrack",
        "SiamONNX",
        "MOSSE (TrackerMOSSE)",
        "KCF (TrackerKCF)",
        "CSRT (TrackerCSRT)",
        "VitTracker (cv2.TrackerVit)",
    ]

    def __init__(self, algorithm: str = "ByteTrack"):
        self.algorithm = algorithm
        self.bytetracker = ByteTracker(high_thresh=0.4, low_thresh=0.1)
        self.siam_tracker = SiamONNXTracker()
        self.cv_trackers: List[OpenCVTrackerInstance] = []
        self._id_counter = 0

    def set_algorithm(self, algorithm: str):
        if algorithm != self.algorithm:
            self.algorithm = algorithm
            self.cv_trackers.clear()

    def update_with_detections(self, frame: Optional[np.ndarray], detections: List[List[Any]]) -> List[List[Any]]:
        """Called when new VLM detection predictions arrive."""
        algo = self.algorithm.upper()

        if "BYTETRACK" in algo:
            formatted_dets = []
            for b in detections:
                if len(b) >= 4:
                    lbl = b[4] if len(b) >= 5 else ""
                    formatted_dets.append([b[0], b[1], b[2], b[3], lbl, 0.9])
            return self.bytetracker.update(formatted_dets)

        elif "SIAM" in algo:
            if frame is not None:
                return self.siam_tracker.init_tracks(frame, detections)

        # OpenCV Tracker family: MOSSE, KCF, CSRT, VitTracker
        self.cv_trackers.clear()
        results = []
        if frame is not None and frame.size > 0:
            for b in detections:
                if len(b) >= 4:
                    ymin, xmin, ymax, xmax = [float(v) for v in b[:4]]
                    label = str(b[4]) if len(b) >= 5 else ""
                    self._id_counter += 1
                    inst = OpenCVTrackerInstance(
                        algo_name=self.algorithm,
                        frame=frame,
                        bbox=(ymin, xmin, ymax, xmax),
                        label=label,
                        track_id=self._id_counter,
                    )
                    if inst.is_valid:
                        self.cv_trackers.append(inst)
                        results.append([ymin, xmin, ymax, xmax, label, self._id_counter])
        return results if results else detections

    def track_frame_only(self, frame: Optional[np.ndarray], last_boxes: List[List[Any]]) -> List[List[Any]]:
        """Called on stream ticks between VLM detections."""
        algo = self.algorithm.upper()

        if "BYTETRACK" in algo:
            return list(last_boxes)

        elif "SIAM" in algo:
            if frame is not None and self.siam_tracker.active_tracks:
                siam_res = self.siam_tracker.track_only(frame)
                if siam_res:
                    return siam_res
            return list(last_boxes)

        # OpenCV Tracker family: MOSSE, KCF, CSRT, VitTracker
        if frame is not None and self.cv_trackers:
            cv_results = []
            for inst in self.cv_trackers:
                box = inst.update(frame)
                if box is not None:
                    cv_results.append(box)
            if cv_results:
                return cv_results

        return list(last_boxes)
