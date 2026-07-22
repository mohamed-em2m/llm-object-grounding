"""
Real-time webcam streaming & video frame detection tab UI and processing functions.

Fixes vs. previous version:
  1. Per-session state (was: module-level globals shared across every connected user).
     Each browser session now gets its own SessionDetector via gr.State, so two people
     streaming at once no longer stomp on each other's boxes / detecting flag.
  2. Detections are frame-correlated with a monotonic frame_id, so a slow detection
     that finishes late can never overwrite a newer result ("flicker back" bug).
  3. Video-file processing now waits for each frame's detection to actually complete
     (bounded via Future.result()) instead of firing a background thread and grabbing
     whatever _last_boxes happens to contain. This was the "results after model ended"
     bug: the loop would move to the next frame before the previous detection thread
     had written its result, and the very last frame(s)' detections were dropped
     entirely because the thread finished after process_video_frames had returned.
  4. OpenAI client + ObjectDetectionPipeline are cached per (base_url, api_key, model)
     instead of being rebuilt on every single tick.
  5. ThreadPoolExecutor(max_workers=1) + Future replaces the raw Thread + bool flag,
     which was a race: `_is_detecting` could be read stale between the check and set.
"""

import io
import time
import html
import itertools
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Any, Optional, List, Tuple

import gradio as gr
from PIL import Image
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from openai import OpenAI
from free_detection.detection_pipeline import (
    ObjectDetectionPipeline,
    pil_to_data_uri,
    parse_detections,
    validate_detections,
)
from free_detection.image_preprocessing import (
    preprocess_resolution,
    map_bbox_to_original,
)

DEFAULT_HUD = '<div class="neo-retro-hud-stat">STATUS: INITIALIZED</div>'

# ---------------------------------------------------------------------------
# Client cache (safe to share across sessions -- it's stateless/read-only,
# unlike the detection results which must stay per-session)
# ---------------------------------------------------------------------------
_client_cache: Dict[Tuple[str, str, str], "ObjectDetectionPipeline"] = {}
_client_cache_lock = Lock()


def _get_pipeline(
    base_url: str, api_key: str, model_name: str
) -> ObjectDetectionPipeline:
    key = (base_url, api_key, model_name)
    with _client_cache_lock:
        pipeline = _client_cache.get(key)
        if pipeline is None:
            client = OpenAI(base_url=base_url, api_key=api_key)
            pipeline = ObjectDetectionPipeline(client=client, detector_model=model_name)
            _client_cache[key] = pipeline
        return pipeline


# ---------------------------------------------------------------------------
# Per-session detection state
# ---------------------------------------------------------------------------
@dataclass
class SessionDetector:
    """
    One instance of this lives inside a gr.State for each connected browser
    session, so concurrent users never share detection results or the
    in-flight flag.
    """

    lock: Lock = field(default_factory=Lock)
    executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="det"
        )
    )
    future: Optional[Future] = None
    last_boxes: List[Any] = field(default_factory=list)
    last_hud: str = DEFAULT_HUD
    last_applied_frame_id: int = -1
    _frame_counter: "itertools.count" = field(
        default_factory=lambda: itertools.count(1)
    )

    def next_frame_id(self) -> int:
        return next(self._frame_counter)

    def is_busy(self) -> bool:
        with self.lock:
            return self.future is not None and not self.future.done()

    def submit(self, frame_id, fn, *args) -> None:
        with self.lock:
            self.future = self.executor.submit(self._run_and_store, frame_id, fn, *args)

    def _run_and_store(self, frame_id, fn, *args):
        try:
            boxes, hud = fn(*args)
        except Exception as e:  # noqa: BLE001
            boxes, hud = None, (
                f'<div class="neo-retro-hud-stat" style="color:#ff0055 !important;">'
                f"ERROR: {html.escape(str(e))}</div>"
            )
        with self.lock:
            # Only apply if this frame is newer than whatever we last applied.
            # Prevents a slow, stale detection from clobbering a fresher result.
            if frame_id >= self.last_applied_frame_id:
                self.last_applied_frame_id = frame_id
                if boxes is not None:
                    self.last_boxes = boxes
                self.last_hud = hud

    def snapshot(self):
        with self.lock:
            return list(self.last_boxes), self.last_hud

    def shutdown(self):
        self.executor.shutdown(wait=False, cancel_futures=True)


def new_session_detector() -> SessionDetector:
    return SessionDetector()


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_boxes_opencv(image_np: np.ndarray, boxes: List[Any]) -> np.ndarray:
    """Fast OpenCV bounding box rendering (<1ms vs ~200ms for Matplotlib)."""
    if cv2 is None or not boxes:
        return image_np
    img = image_np.copy()
    for box in boxes:
        if len(box) >= 4:
            ymin, xmin, ymax, xmax = box[:4]
            label = str(box[4]) if len(box) >= 5 else ""
            pt1 = (int(xmin), int(ymin))
            pt2 = (int(xmax), int(ymax))
            # Vibrant cyber cyan bounding box (#00ffcc -> BGR (204, 255, 0))
            cv2.rectangle(img, pt1, pt2, (204, 255, 0), 2)
            if label:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_w, text_h), _ = cv2.getTextSize(
                    label, font, font_scale, thickness
                )
                lbl_pt1 = (int(xmin), max(0, int(ymin) - text_h - 6))
                lbl_pt2 = (int(xmin) + text_w + 6, max(text_h + 6, int(ymin)))
                cv2.rectangle(img, lbl_pt1, lbl_pt2, (204, 255, 0), -1)
                cv2.putText(
                    img,
                    label,
                    (int(xmin) + 3, max(text_h + 2, int(ymin) - 3)),
                    font,
                    font_scale,
                    (17, 8, 5),
                    thickness,
                    cv2.LINE_AA,
                )
    return img


# ---------------------------------------------------------------------------
# Core detection call (pure -- no globals, no gradio-specific side effects)
# ---------------------------------------------------------------------------
def _detect(
    frame: np.ndarray,
    categories: list,
    base_url: str,
    api_key: str,
    model_name: str,
    prep_info: dict,
) -> Tuple[List[Any], str]:
    """Runs VLM detection on a single (already-downscaled) frame and returns
    (boxes_in_original_pixel_space, hud_html). Raises on failure -- caller
    is responsible for catching."""
    start_time = time.time()
    pil_img = Image.fromarray(frame).convert("RGB")
    pipeline = _get_pipeline(base_url, api_key, model_name)
    img_uri = pil_to_data_uri(pil_img)
    raw_output = pipeline.run_inference(
        image_uris=img_uri,
        categories=categories,
        category_definitions="",
    )
    parsed_dets = parse_detections(raw_output)
    valid_dets = validate_detections(parsed_dets, categories)

    orig_w = prep_info["orig_w"]
    orig_h = prep_info["orig_h"]
    boxes = []
    for d in valid_dets:
        bbox = d.get("bbox_2d", [])
        lbl = d.get("label", "")
        if len(bbox) == 4:
            x1, y1, x2, y2 = map_bbox_to_original(list(bbox), prep_info)
            ymin = y1 * orig_h / 1000.0
            xmin = x1 * orig_w / 1000.0
            ymax = y2 * orig_h / 1000.0
            xmax = x2 * orig_w / 1000.0
            boxes.append([ymin, xmin, ymax, xmax, lbl])

    elapsed = (time.time() - start_time) * 1000.0
    fps = 1000.0 / max(elapsed, 1.0)
    hud = (
        f'<div class="neo-retro-hud-stat">FPS: {fps:.1f} | '
        f"LATENCY: {elapsed:.0f}ms | DETECTED: {len(boxes)}</div>"
    )
    return boxes, hud


def _resolve_endpoint(
    server_port, use_external_api, ext_api_url, ext_api_key, ext_model_name
):
    base_url = ext_api_url if use_external_api else f"http://127.0.0.1:{server_port}/v1"
    api_key = ext_api_key if use_external_api else "no-key"
    model_name = ext_model_name if use_external_api else "local-model"
    return base_url, api_key, model_name


# ---------------------------------------------------------------------------
# Webcam streaming tick (async / non-blocking -- must never stall the UI)
# ---------------------------------------------------------------------------
def process_single_frame(
    frame: np.ndarray,
    categories_str: str,
    server_port: int,
    use_external_api: bool,
    ext_api_url: str,
    ext_api_key: str,
    ext_model_name: str,
    confidence_thresh: float,
    max_resolution: int,
    session: SessionDetector,
) -> tuple[Optional[np.ndarray], str, SessionDetector]:
    """
    Called on every Gradio webcam stream tick.
    - Every tick receives the LIVE webcam frame from the browser.
    - If no background detection is running *for this session*, launch one
      asynchronously on a downscaled frame.
    - Render the latest known boxes for this session onto the CURRENT LIVE frame.
    - The stream stays smooth while boxes update asynchronously, and results
      from different sessions can never mix.
    """
    if session is None:
        session = new_session_detector()

    if frame is None:
        _, hud = session.snapshot()
        return None, hud, session

    pil_img = Image.fromarray(frame).convert("RGB")
    max_res = int(max_resolution or 640)
    proc_img, prep_info = preprocess_resolution(
        pil_img, enabled=True, target_short_edge=max_res
    )

    categories = [c.strip() for c in categories_str.split(",") if c.strip()] or [
        "object"
    ]
    base_url, api_key, model_name = _resolve_endpoint(
        server_port, use_external_api, ext_api_url, ext_api_key, ext_model_name
    )

    if not session.is_busy():
        frame_id = session.next_frame_id()
        proc_np = np.array(proc_img)
        session.submit(
            frame_id,
            _detect,
            proc_np,
            categories,
            base_url,
            api_key,
            model_name,
            prep_info,
        )

    boxes_to_draw, hud = session.snapshot()
    annotated_live = draw_boxes_opencv(np.array(pil_img), boxes_to_draw)
    return annotated_live, hud, session


def reset_session(session: Optional[SessionDetector]) -> SessionDetector:
    """Call when the stream stops / mode toggles, so a stale in-flight
    detection from the old context can't leak into the new one."""
    if session is not None:
        session.shutdown()
    return new_session_detector()


# ---------------------------------------------------------------------------
# Video file processing (SYNCHRONOUS per sampled frame -- deliberately not
# fire-and-forget, so every sampled frame's result is guaranteed to be
# captured before the function returns, and progress reporting stays honest)
# ---------------------------------------------------------------------------
def process_video_frames(
    video_path: str,
    sample_interval: float,
    categories_str: str,
    server_port: int,
    use_external_api: bool,
    ext_api_url: str,
    ext_api_key: str,
    ext_model_name: str,
    max_resolution: int = 640,
    progress=gr.Progress(),
) -> tuple[List[np.ndarray], str]:
    if not video_path:
        return [], "No video file uploaded."
    if cv2 is None:
        return [], "OpenCV (cv2) is required for video processing."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], "Failed to open video file."

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = int(max(1, fps * sample_interval))
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_count += 1
    cap.release()

    if not frames:
        return [], "No frames could be sampled from this video."

    categories = [c.strip() for c in categories_str.split(",") if c.strip()] or [
        "object"
    ]
    base_url, api_key, model_name = _resolve_endpoint(
        server_port, use_external_api, ext_api_url, ext_api_key, ext_model_name
    )
    max_res = int(max_resolution or 640)

    annotated_frames = []
    errors = 0
    for idx, f in enumerate(frames):
        progress(
            (idx + 1) / len(frames), desc=f"Detecting frame {idx + 1}/{len(frames)}"
        )
        pil_img = Image.fromarray(f).convert("RGB")
        proc_img, prep_info = preprocess_resolution(
            pil_img, enabled=True, target_short_edge=max_res
        )
        try:
            # Blocking call -- we WAIT for this frame's result before moving on,
            # so nothing is ever dropped or shifted relative to the frame it came from.
            boxes, _hud = _detect(
                np.array(proc_img), categories, base_url, api_key, model_name, prep_info
            )
        except Exception:
            boxes = []
            errors += 1
        annotated_frames.append(draw_boxes_opencv(np.array(pil_img), boxes))

    status = f"Successfully processed {len(annotated_frames)} frames from video!"
    if errors:
        status += f" ({errors} frame(s) failed detection and were shown unannotated.)"
    return annotated_frames, status


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def _build_realtime_tab() -> Dict[str, Any]:
    c = {}
    # Per-session detection state -- NOT shared between browser tabs/users.
    c["session_state"] = gr.State(new_session_detector)

    with gr.Column(elem_classes=["neo-retro-card"]):
        gr.HTML(
            """
        <div style="padding: 10px; border-bottom: 2px solid #00ffcc; background: #050811;">
            <span class="neo-retro-badge">LIVE CYBER-STREAM</span>
            <h2 style="color: #00ffcc; font-family: 'JetBrains Mono', monospace; margin: 5px 0 0;">
                ⚡ REAL-TIME WEBCAM & VIDEO FRAME DETECTOR
            </h2>
        </div>
        """
        )
        with gr.Row():
            with gr.Column(scale=1):
                c["stream_mode"] = gr.Radio(
                    choices=["Webcam Stream", "Video Upload (1s Sampling)"],
                    value="Webcam Stream",
                    label="STREAM INPUT SOURCE",
                )
                c["categories_input"] = gr.Textbox(
                    value="person, car, dog, bottle, phone",
                    label="TARGET CATEGORIES (comma-separated)",
                )
                c["max_resolution"] = gr.Slider(
                    minimum=384,
                    maximum=1084,
                    step=128,
                    value=640,
                    label="MAX FRAME RESOLUTION (PX)",
                    info="Lower resolution = faster real-time processing and lower latency.",
                )
                c["sample_interval"] = gr.Slider(
                    minimum=0.5,
                    maximum=5.0,
                    step=0.5,
                    value=1.0,
                    label="VIDEO FRAME SAMPLING INTERVAL (SECONDS)",
                )
                c["process_video_btn"] = gr.Button(
                    "⚡ PROCESS VIDEO FRAMES",
                    variant="primary",
                    elem_classes=["neo-retro-badge"],
                )
                c["hud_status"] = gr.HTML(value=DEFAULT_HUD)
            with gr.Column(scale=2):
                c["webcam_input"] = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    label="LIVE WEBCAM STREAM",
                    type="numpy",
                )
                c["video_input"] = gr.Video(
                    label="INPUT VIDEO FILE",
                    visible=False,
                )
                c["annotated_stream_output"] = gr.Image(
                    label="NEO-RETRO DETECTED STREAM / OVERLAY",
                    type="numpy",
                )
                c["video_gallery_output"] = gr.Gallery(
                    label="SAMPLED FRAME DETECTIONS (EVERY 1 SEC)",
                    visible=False,
                    columns=3,
                )
    return c


def _wire_realtime_events(
    c_real: Dict[str, Any], c_srv: Dict[str, Any], c_bat: Dict[str, Any]
):
    def toggle_mode(mode, session):
        is_cam = mode == "Webcam Stream"
        # Tear down any in-flight detection from the mode we're leaving so it
        # can't write stale results into the mode we're entering.
        fresh_session = reset_session(session)
        return (
            gr.update(visible=is_cam),
            gr.update(visible=not is_cam),
            gr.update(visible=is_cam),
            gr.update(visible=not is_cam),
            fresh_session,
        )

    c_real["stream_mode"].change(
        toggle_mode,
        inputs=[c_real["stream_mode"], c_real["session_state"]],
        outputs=[
            c_real["webcam_input"],
            c_real["video_input"],
            c_real["annotated_stream_output"],
            c_real["video_gallery_output"],
            c_real["session_state"],
        ],
    )

    c_real["webcam_input"].stream(
        fn=process_single_frame,
        inputs=[
            c_real["webcam_input"],
            c_real["categories_input"],
            c_srv["server_port_input"],
            c_bat["use_external_api_chk"],
            c_bat["ext_api_url"],
            c_bat["ext_api_key"],
            c_bat["ext_model_name"],
            gr.State(0.3),
            c_real["max_resolution"],
            c_real["session_state"],
        ],
        outputs=[
            c_real["annotated_stream_output"],
            c_real["hud_status"],
            c_real["session_state"],
        ],
        stream_every=0.3,
    )

    c_real["process_video_btn"].click(
        fn=process_video_frames,
        inputs=[
            c_real["video_input"],
            c_real["sample_interval"],
            c_real["categories_input"],
            c_srv["server_port_input"],
            c_bat["use_external_api_chk"],
            c_bat["ext_api_url"],
            c_bat["ext_api_key"],
            c_bat["ext_model_name"],
            c_real["max_resolution"],
        ],
        outputs=[c_real["video_gallery_output"], c_real["hud_status"]],
    )
