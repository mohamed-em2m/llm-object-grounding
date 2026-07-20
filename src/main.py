from .auto_annotation import main as aa_main
from .free_detection import main as free_detection
import argparse

from .schemes import PipelineConfig

"""
Unified configuration for the VLM defect-relabeling + LLM object-detection pipeline.

Exposes:
  * PipelineConfig   – a pydantic model that validates every CLI option.
  * build_parser()   – merged argparse parser mirroring PipelineConfig.
  * parse_args()     – convenience helper that returns a PipelineConfig instance.
"""

from typing import List, Optional, Literal

import argparse
import yaml

# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Argparse parser (mirrors PipelineConfig exactly)
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vlm-pipeline",
        description=(
            "Unified CLI for (a) relabeling binary defect/no-defect YOLO annotations "
            "into multi-class defect labels with a VLM, and (b) running the LLM "
            "object-detection pipeline on individual images."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Logging -----------------------------------------------------------
    p.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level.",
    )
    p.add_argument("--log_file", default=None, help="Optional log file path.")

    # --- Input images ------------------------------------------------------
    p.add_argument(
        "--image",
        "-i",
        metavar="PATH",
        action="append",
        dest="images",
        help="Path to an input image (detection mode). Repeatable.",
    )
    p.add_argument(
        "--train_image", default=None, help="Folder of training images (relabel mode)."
    )
    p.add_argument(
        "--train_label",
        default=None,
        help="Folder of YOLO training labels (relabel mode).",
    )
    p.add_argument(
        "--yaml_path", default=None, help="Path to dataset YAML (data.yaml)."
    )
    p.add_argument(
        "--image_extensions",
        default=".jpg,.jpeg,.png",
        help="Comma-separated image extensions to process.",
    )

    # --- Categories --------------------------------------------------------
    p.add_argument(
        "--categories",
        "-c",
        default="person, car, bicycle, dog, cat",
        help="Comma-separated object categories to detect.",
    )
    p.add_argument(
        "--definitions",
        "-d",
        default="",
        help="Optional category definitions (plain text, one per line).",
    )
    p.add_argument(
        "--init_class_map",
        action="store_true",
        help="Initialize class map from the YAML file.",
    )
    p.add_argument(
        "--conf_threshold",
        type=int,
        default=2,
        choices=[1, 2, 3, 4, 5],
        help="Confidence threshold (1–5) for low-confidence review log.",
    )

    # --- Output ------------------------------------------------------------
    p.add_argument(
        "--output_folder",
        "-o",
        default="./detection_results",
        help="Output directory (results or relabeled annotations).",
    )
    p.add_argument(
        "--inplace_saving",
        action="store_true",
        help="Save relabeled annotations in the original label folder.",
    )
    p.add_argument(
        "--no_plot", action="store_true", help="Skip the matplotlib preview window."
    )

    # --- Sampling / slicing ------------------------------------------------
    p.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Only process this many images (sanity check).",
    )
    p.add_argument(
        "--shuffle", action="store_true", help="Shuffle image order (seeded by --seed)."
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="Start index (0-based, inclusive).",
    )
    p.add_argument(
        "--end_index", type=int, default=None, help="End index (0-based, exclusive)."
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Images per batch (0 disables batching).",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Don't call model or write files; just print plan.",
    )

    # --- Resume ------------------------------------------------------------
    p.add_argument(
        "--resume", action="store_true", help="Legacy per-file resume check."
    )
    p.add_argument(
        "--auto_resume",
        action="store_true",
        default=True,
        help="Resume from <output_folder>/.checkpoint.json (default on).",
    )
    p.add_argument(
        "--no_auto_resume",
        action="store_false",
        dest="auto_resume",
        help="Disable auto-resume and start fresh.",
    )

    # --- Server / model ----------------------------------------------------
    p.add_argument(
        "--model",
        default="local-model",
        help="Model name (relabeler / single-model mode).",
    )
    p.add_argument(
        "--detector_model",
        default="local-model",
        help="Detector model name sent in API request.",
    )
    p.add_argument(
        "--judge_model",
        default="local-model",
        help="Judge model name sent in API request.",
    )
    p.add_argument(
        "--judge_url", default=None, help="Separate base URL for the judge model."
    )
    p.add_argument("--api_key", default="not-needed", help="API key.")
    p.add_argument(
        "--base_url",
        default="http://localhost:8080/v1",
        help="OpenAI-compatible base URL of the inference server.",
    )
    p.add_argument(
        "--server_type",
        default="llama_cpp",
        choices=["llama_cpp", "vllm", "external"],
        help="Server backend to use.",
    )
    p.add_argument(
        "--max_workers", type=int, default=1, help="Concurrent image workers."
    )

    # llama.cpp
    p.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable model thinking/reasoning on llama.cpp.",
    )
    p.add_argument(
        "--use_mtp",
        action="store_true",
        default=True,
        help="Enable draft-MTP speculative decoding (llama.cpp).",
    )
    p.add_argument(
        "--no_mtp",
        action="store_false",
        dest="use_mtp",
        help="Disable draft-MTP speculative decoding.",
    )

    p.add_argument(
        "--serving_extra",
        type=dict,
        default={},
        help="this argument will used in running and launagch server",
    )
    p.add_argument(
        "--ctx_size", type=int, default=20000, help="Context size for llama.cpp."
    )
    p.add_argument("--port", type=int, default=8080, help="Port for llama.cpp server.")
    p.add_argument(
        "--parallel_slots",
        type=int,
        default=1,
        help="Parallel inference slots for llama.cpp.",
    )

    # --- Detection pipeline tuning ----------------------------------------
    p.add_argument(
        "--max_rounds", type=int, default=2, help="Max detector/judge rounds per image."
    )
    p.add_argument(
        "--score_threshold",
        type=int,
        default=8,
        help="Stop early when judge score reaches this value (0–10).",
    )
    p.add_argument("--detector_temperature", type=float, default=0.9)
    p.add_argument("--detector_top_p", type=float, default=0.95)
    p.add_argument("--judge_temperature", type=float, default=0.2)
    p.add_argument("--detector_max_tokens", type=int, default=4096)
    p.add_argument("--judge_max_tokens", type=int, default=1024)
    p.add_argument("--api_retries", type=int, default=3)

    # --- vLLM configuration ------------------------------------------------
    p.add_argument("--max_model_len", type=int, default=20000)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--pipeline_parallel_size", type=int, default=1)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--quantization", default=None)
    p.add_argument("--kv_cache_dtype", default="auto")
    p.add_argument("--max_num_seqs", type=int, default=2)
    p.add_argument("--enforce_eager", action="store_true")
    p.add_argument("--enable_chunked_prefill", action="store_true", default=True)
    p.add_argument(
        "--no_chunked_prefill", action="store_false", dest="enable_chunked_prefill"
    )
    p.add_argument("--enable_prefix_caching", action="store_true", default=True)
    p.add_argument(
        "--no_prefix_caching", action="store_false", dest="enable_prefix_caching"
    )
    p.add_argument("--speculative_model", default=None)
    p.add_argument("--num_speculative_tokens", type=int, default=None)
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    p.add_argument(
        "--no_trust_remote_code", action="store_false", dest="trust_remote_code"
    )
    p.add_argument("--download_dir", default=None)
    p.add_argument("--limit_mm_per_prompt", default=None)
    p.add_argument("--chat_template", default=None)
    p.add_argument(
        "--extra_args",
        action="append",
        default=None,
        help="Extra args forwarded to vLLM server.",
    )

    # --- VLM image encoding -----------------------------------------------
    p.add_argument("--image_min_tokens", type=int, default=1024)
    p.add_argument("--image_max_tokens", type=int, default=4096)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)

    # --- Preprocessing -----------------------------------------------------
    p.add_argument(
        "--prep_enabled", action="store_true", help="Enable image preprocessing."
    )
    p.add_argument("--prep_short_edge", type=int, default=1024)
    p.add_argument("--prep_pad_square", action="store_true")
    p.add_argument(
        "--prep_contrast_method",
        choices=["none", "clahe", "autocontrast"],
        default="none",
    )
    p.add_argument("--prep_gamma", type=float, default=1.0)
    p.add_argument(
        "--prep_denoise_method", choices=["none", "bilateral", "nlm"], default="none"
    )
    p.add_argument("--prep_sharpen", action="store_true")
    p.add_argument("--prep_white_balance", action="store_true")
    p.add_argument(
        "--prep_grid_style",
        choices=["standard", "transparent", "fine", "none"],
        default="standard",
    )
    p.add_argument(
        "--prep_som_enabled",
        action="store_true",
        help="Enable Set-of-Mark visual prompting overlay.",
    )
    p.add_argument("--prep_tiling_enabled", action="store_true")
    p.add_argument("--prep_tile_size", type=int, default=512)
    p.add_argument("--prep_tile_overlap", type=float, default=0.2)
    p.add_argument("--prep_crop_verify_enabled", action="store_true")
    p.add_argument("--prep_crop_padding", type=float, default=0.15)

    # Custom grid overlays
    p.add_argument("--prep_grid_step", type=int, default=100)
    p.add_argument("--prep_grid_line_width", type=int, default=1)
    p.add_argument("--prep_grid_font_size", type=int, default=0)
    p.add_argument("--prep_grid_line_color", default="red")
    p.add_argument("--prep_grid_text_color", default="white")
    p.add_argument("--prep_grid_backing_color", default="black")

    # VLM processor pixels
    p.add_argument("--prep_send_pixel_bounds", action="store_true")
    p.add_argument("--prep_min_pixels", type=int, default=200_704)
    p.add_argument("--prep_max_pixels", type=int, default=4_194_304)

    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to PipelineConfig yaml file.",
    )
    return p


# ---------------------------------------------------------------------------
# parse_args helper
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> PipelineConfig:
    """Parse CLI args and return a validated :class:`PipelineConfig`."""

    parser = build_parser()

    # Load config from YAML if provided
    if args.config:
        with open(args.config, "r") as f:
            cfg_yaml = yaml.safe_load(f)
            # Update ns with YAML values (they will be overwritten by explicit CLI args)
            ns = vars(ns)
            for k, v in cfg_yaml.items():
                if k in ns:
                    ns[k] = v
            ns = argparse.Namespace(**ns)

    ns = parser.parse_args(argv)
    # Drop any None-valued extras so pydantic defaults apply cleanly.
    raw = {k: v for k, v in vars(ns).items() if v is not None}
    try:
        return PipelineConfig(**raw)
    except Exception as exc:  # surface validation errors nicely
        parser.error(str(exc))


if __name__ == "__main__":
    # parse args
    args = parse_args()

    if args.task == "auto_label":
        aa_main(args)
    elif args.task == "free_detection":
        free_detection(args)

    else:
        raise ValueError("Invalid run type")
