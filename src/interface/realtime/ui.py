"""
Gradio UI layout and event wiring for the Realtime streaming tab.
"""

from typing import Dict, Any
import gradio as gr

from free_detection.trackers import MultiAlgorithmTracker
from interface.realtime.state import (
    SessionDetector,
    new_session_detector,
    reset_session,
    DEFAULT_HUD,
)
from interface.realtime.handlers import process_single_frame, process_video_frames


def _build_realtime_tab() -> Dict[str, Any]:
    c = {}
    c["session_state"] = gr.State(new_session_detector)

    with gr.Column(elem_classes=["neo-retro-card"]):
        gr.HTML(
            """
        <div style="padding: 10px; border-bottom: 2px solid #00ffcc; background: #050811;">
            <span class="neo-retro-badge">LIVE CYBER-STREAM</span>
            <h2 style="color: #00ffcc; font-family: 'JetBrains Mono', monospace; margin: 5px 0 0;">
                ⚡ REAL-TIME WEBCAM & VIDEO FRAME DETECTOR (MULTI-TRACKER INTEGRATED)
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
                c["tracker_algorithm"] = gr.Dropdown(
                    choices=MultiAlgorithmTracker.SUPPORTED_ALGOS,
                    value="CSRT (TrackerCSRT)",
                    label="REAL-TIME TRACKING ALGORITHM",
                    info=(
                        "None = show raw VLM boxes. "
                        "MOSSE/KCF/CSRT/VitTracker = OpenCV single-object trackers "
                        "that propagate boxes between VLM calls. "
                        "ByteTrack = multi-object Kalman IoU tracker."
                    ),
                )
                c["categories_input"] = gr.Textbox(
                    value="person, car, dog, bottle, phone",
                    label="TARGET CATEGORIES (comma-separated)",
                    info="Leave empty or type * for free/open-vocabulary detection.",
                )

                # ── Resolution ───────────────────────────────────────────────
                c["enable_resizing"] = gr.Checkbox(
                    value=True,
                    label="ENABLE IMAGE RESIZING",
                    info="Uncheck to pass native resolution to VLM.",
                )
                c["max_resolution"] = gr.Number(
                    value=1024,
                    label="MAX FRAME RESOLUTION (PX)",
                    info="Target short-edge resolution for the VLM input.",
                    precision=0,
                    visible=True,
                )

                # ── Motion Gate + Refresh ─────────────────────────────────────
                c["motion_gate_enabled"] = gr.Checkbox(
                    value=True,
                    label="⚡ MOTION GATE (Scene-Change Gating)",
                    info="ON: only re-detect when scene changes or stale timer fires. "
                         "OFF: re-detect as fast as GPU can respond (no scene-change check).",
                )
                c["motion_sensitivity"] = gr.Slider(
                    minimum=0.5,
                    maximum=10.0,
                    step=0.5,
                    value=1.5,
                    label="MOTION SENSITIVITY (% PIXELS CHANGED)",
                    info="Lower = more sensitive — more VLM calls.",
                )
                c["stale_refresh"] = gr.Slider(
                    minimum=1.0,
                    maximum=20.0,
                    step=0.5,
                    value=3.0,
                    label="STALE REFRESH FALLBACK (SECONDS)",
                    info="Re-detect anyway after this long even with no motion. Ignored when Motion Gate is OFF.",
                )

                # ── Section A: VLM Image Conditioning ────────────────────────
                with gr.Accordion("🎨 Section A — VLM Image Conditioning", open=False):
                    gr.HTML(
                        '<p style="color:#aaa;font-size:12px;margin:2px 0 8px;">'
                        "Applied to what the model <em>sees</em> — keeps image clean & consistent."
                        "</p>"
                    )
                    c["vlm_conditioning"] = gr.Checkbox(
                        value=True,
                        label="Enable VLM Conditioning Pipeline",
                        info="Master toggle for CLAHE + White Balance + Bilateral Denoise.",
                    )
                    with gr.Group() as c["conditioning_group"]:
                        c["clahe_enabled"] = gr.Checkbox(
                            value=True,
                            label="CLAHE Contrast Normalization",
                            info="Normalizes uneven lighting (best single win for consistency).",
                        )
                        c["clahe_clip"] = gr.Slider(
                            minimum=0.5, maximum=8.0, step=0.5, value=2.0,
                            label="CLAHE Clip Limit",
                            info="Higher = stronger local contrast boost.",
                        )
                        c["white_balance"] = gr.Checkbox(
                            value=True,
                            label="Gray World White Balance",
                            info="Corrects color temperature drift between shots.",
                        )
                        c["denoise_method"] = gr.Dropdown(
                            choices=["bilateral", "nlm", "none"],
                            value="bilateral",
                            label="Denoise Method",
                            info="Bilateral preserves defect edges; NLM is stronger but slower.",
                        )
                        c["denoise_d"] = gr.Slider(
                            minimum=3, maximum=15, step=2, value=5,
                            label="Bilateral Filter Diameter (d)",
                            info="Larger = stronger smoothing. Keep ≤7 for real-time speed.",
                        )

                    gr.HTML('<hr style="border-color:#444;margin:8px 0;">')
                    gr.HTML(
                        '<p style="color:#aaa;font-size:12px;margin:2px 0 6px;">'
                        "<b>Pipeline Parity</b> — steps 3 &amp; 4 from detection_pipeline.py"
                        "</p>"
                    )
                    c["contrast_method"] = gr.Dropdown(
                        choices=["none", "clahe", "gamma", "autocontrast"],
                        value="none",
                        label="Contrast Enhancement Method",
                        info="'clahe' = adaptive equalization; 'gamma' = brightness curve; 'none' = skip.",
                    )
                    c["gamma"] = gr.Number(
                        value=1.0,
                        label="Gamma Value",
                        info="< 1.0 = brighten shadows; > 1.0 = darken. Used only when method = gamma.",
                        precision=2,
                    )
                    c["noise_method"] = gr.Dropdown(
                        choices=["none", "bilateral", "nlm", "gaussian"],
                        value="none",
                        label="Noise Filter Method (Step 4)",
                        info="Applied after contrast. 'none' skips; 'bilateral' best for edge detail.",
                    )
                    c["sharpen"] = gr.Checkbox(
                        value=False,
                        label="Unsharp Mask Sharpening",
                        info="Enhances edges after denoising. Useful for crisp defect boundaries.",
                    )

                # ── Section B: Pre-Filter Triage ─────────────────────────────
                with gr.Accordion("🔎 Section B — Pre-Filter Triage (Fast CV)", open=False):
                    gr.HTML(
                        '<p style="color:#aaa;font-size:12px;margin:2px 0 8px;">'
                        "Runs <em>before</em> VLM — reject bad frames fast to save tokens."
                        "</p>"
                    )
                    c["triage_enabled"] = gr.Checkbox(
                        value=True,
                        label="Enable Pre-Filter Triage",
                        info="Master toggle for all triage heuristics below.",
                    )
                    with gr.Group() as c["triage_group"]:
                        c["blur_reject"] = gr.Checkbox(
                            value=True,
                            label="Blur Rejection (Laplacian Variance)",
                            info="Rejects out-of-focus frames before VLM call.",
                        )
                        c["blur_laplacian_min"] = gr.Slider(
                            minimum=5.0, maximum=150.0, step=5.0, value=30.0,
                            label="Min Laplacian Variance (blur threshold)",
                            info="Frames with var < this are rejected as blurry.",
                        )
                        c["edge_triage"] = gr.Checkbox(
                            value=True,
                            label="Canny Edge Density Trigger",
                            info="Trigger VLM if unusual edge structure detected.",
                        )
                        c["edge_density_thresh"] = gr.Slider(
                            minimum=0.005, maximum=0.2, step=0.005, value=0.02,
                            label="Edge Density Threshold",
                        )
                        c["entropy_triage"] = gr.Checkbox(
                            value=True,
                            label="Local Texture Entropy Trigger",
                            info="Detects texture anomalies (holes, snags, stains).",
                        )
                        c["entropy_variance_thresh"] = gr.Slider(
                            minimum=0.5, maximum=20.0, step=0.5, value=2.0,
                            label="Entropy Variance Threshold",
                        )
                        c["ref_triage"] = gr.Checkbox(
                            value=True,
                            label="Difference-from-Reference Trigger",
                            info="Trigger VLM only when frame differs enough from last submission.",
                        )

                # ── Video / HUD ───────────────────────────────────────────────
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
                # NOTE: Gradio 6.x strips <script> from gr.HTML for security.
                # Only inject the CSS here; all canvas JS runs in the .change(js=) handler.
                gr.HTML(
                    """
                    <style>
                    #rt_float_canvas {
                        position: fixed;
                        pointer-events: none;
                        z-index: 9999;
                        box-sizing: border-box;
                    }
                    </style>
                    """
                )
                with gr.Group(elem_id="rt_webcam_wrap") as webcam_wrap:
                    c["webcam_input"] = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        label="LIVE WEBCAM STREAM",
                        type="numpy",
                        elem_id="rt_webcam_input",
                    )
                c["webcam_wrap_group"] = webcam_wrap
                c["boxes_json_state"] = gr.JSON(visible=False)
                c["video_input"] = gr.Video(
                    label="INPUT VIDEO FILE",
                    visible=False,
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
    c_real["enable_resizing"].change(
        fn=lambda enabled: gr.update(visible=enabled),
        inputs=[c_real["enable_resizing"]],
        outputs=[c_real["max_resolution"]],
    )

    def toggle_mode(mode, session):
        is_cam = mode == "Webcam Stream"
        fresh_session = reset_session(session)
        return (
            gr.update(visible=is_cam),
            gr.update(visible=not is_cam),
            gr.update(visible=not is_cam),
            fresh_session,
        )

    c_real["stream_mode"].change(
        toggle_mode,
        inputs=[c_real["stream_mode"], c_real["session_state"]],
        outputs=[
            c_real["webcam_wrap_group"],
            c_real["video_input"],
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
            gr.State(0.3),               # confidence_thresh (unused, kept for compat)
            c_real["enable_resizing"],
            c_real["max_resolution"],
            c_real["motion_sensitivity"],
            c_real["stale_refresh"],
            c_real["tracker_algorithm"],
            c_real["session_state"],
            # Motion gate master toggle
            c_real["motion_gate_enabled"],
            # Section A — VLM Conditioning
            c_real["vlm_conditioning"],
            c_real["clahe_enabled"],
            c_real["clahe_clip"],
            c_real["white_balance"],
            c_real["denoise_method"],
            c_real["denoise_d"],
            # Pipeline parity — Contrast (step 3) + Noise/Sharpen (step 4)
            c_real["contrast_method"],
            c_real["gamma"],
            c_real["noise_method"],
            c_real["sharpen"],
            # Section B — Triage
            c_real["triage_enabled"],
            c_real["blur_reject"],
            c_real["blur_laplacian_min"],
            c_real["edge_triage"],
            c_real["edge_density_thresh"],
            c_real["entropy_triage"],
            c_real["entropy_variance_thresh"],
            c_real["ref_triage"],
        ],
        outputs=[
            c_real["boxes_json_state"],
            c_real["hud_status"],
            c_real["session_state"],
        ],
        stream_every=0.1,
    )

    c_real["boxes_json_state"].change(
        fn=None,
        inputs=[c_real["boxes_json_state"]],
        outputs=[],
        js="""
        (payload) => {
            if (!payload) return;

            // ── 1. Lazily create the floating canvas once (gr.HTML strips <script>) ──
            var canvas = document.getElementById('rt_float_canvas');
            if (!canvas) {
                canvas = document.createElement('canvas');
                canvas.id = 'rt_float_canvas';
                canvas.style.position      = 'fixed';
                canvas.style.pointerEvents = 'none';
                canvas.style.zIndex        = '9999';
                canvas.style.boxSizing     = 'border-box';
                document.body.appendChild(canvas);
            }

            // ── 2. Start the rAF position loop exactly once ───────────────────────
            if (!window._rtCanvasLoopRunning) {
                window._rtCanvasLoopRunning = true;
                (function loop() {
                    var cv = document.getElementById('rt_float_canvas');
                    if (!cv) { window._rtCanvasLoopRunning = false; return; }
                    var anchor = document.getElementById('rt_webcam_input');
                    if (anchor) {
                        var vid = anchor.querySelector('video');
                        var target = vid || anchor;
                        var r = target.getBoundingClientRect();
                        if (r.width > 0 && r.height > 0) {
                            cv.style.left   = r.left   + 'px';
                            cv.style.top    = r.top    + 'px';
                            cv.style.width  = r.width  + 'px';
                            cv.style.height = r.height + 'px';
                            var rw = Math.round(r.width),  rh = Math.round(r.height);
                            if (cv.width !== rw)  cv.width  = rw;
                            if (cv.height !== rh) cv.height = rh;
                        }
                    }
                    requestAnimationFrame(loop);
                })();
            }

            // ── 3. Draw boxes ─────────────────────────────────────────────────────
            var ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            var boxes  = payload.boxes  || [];
            var frameW = payload.frame_w || canvas.width;
            var frameH = payload.frame_h || canvas.height;
            if (!frameW || !frameH || boxes.length === 0) return;

            var scaleX = canvas.width  / frameW;
            var scaleY = canvas.height / frameH;

            ctx.lineWidth = 2;
            ctx.font = '12px "JetBrains Mono", monospace';

            for (var i = 0; i < boxes.length; i++) {
                var box = boxes[i];
                if (!box || box.length < 4) continue;
                var ymin = box[0], xmin = box[1], ymax = box[2], xmax = box[3];
                var label   = box[4] !== undefined ? String(box[4]) : '';
                var trackId = box[5] !== undefined ? box[5] : null;
                var x = xmin * scaleX,  y = ymin * scaleY;
                var w = (xmax - xmin) * scaleX,  h = (ymax - ymin) * scaleY;

                // Neon cyan box with glow
                ctx.strokeStyle = '#00ffcc';
                ctx.shadowColor = '#00ffcc';
                ctx.shadowBlur  = 6;
                ctx.strokeRect(x, y, w, h);
                ctx.shadowBlur  = 0;

                var tag = trackId !== null ? (label + ' #' + trackId) : label;
                if (tag.trim()) {
                    var tw = ctx.measureText(tag).width;
                    var bh = 18;
                    var by = (y > bh) ? (y - bh) : (y + h);
                    ctx.fillStyle = 'rgba(0,255,204,0.85)';
                    ctx.fillRect(x, by, tw + 8, bh);
                    ctx.fillStyle = '#050811';
                    ctx.fillText(tag, x + 4, by + bh - 4);
                }
            }
        }
        """,
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
            c_real["enable_resizing"],
            c_real["max_resolution"],
            c_real["tracker_algorithm"],
        ],
        outputs=[c_real["video_gallery_output"], c_real["hud_status"]],
    )
