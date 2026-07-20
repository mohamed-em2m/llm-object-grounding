from .auto_annotation import main as aa_main
from .free_detection import main as free_detection
import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="detection-cli",
        description="Run the LLM object-detection pipeline on one or more images from the command line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--task",
        type=str,
        default="auto_label",
        choices=["auto_label", "free_detection"],
    )
    # --- Images ---
    p.add_argument(
        "--image",
        "-i",
        metavar="PATH",
        action="append",
        required=True,
        dest="images",
        help="Path to an input image. Can be specified multiple times.",
    )

    # --- Categories ---
    p.add_argument(
        "--categories",
        "-c",
        metavar="LIST",
        default="person, car, bicycle, dog, cat",
        help="Comma-separated list of object categories to detect.",
    )
    p.add_argument(
        "--definitions",
        "-d",
        metavar="TEXT",
        default="",
        help="Optional category definitions (plain text, one per line).",
    )

    # --- Server / model ---
    p.add_argument(
        "--base-url",
        metavar="URL",
        default="http://localhost:8080/v1",
        help="OpenAI-compatible base URL of the inference server.",
    )
    p.add_argument(
        "--api-key",
        metavar="KEY",
        default="not-needed",
        help="API key (use 'not-needed' for local servers).",
    )
    p.add_argument(
        "--detector-model",
        metavar="NAME",
        default="local-model",
        help="Model name sent in the detector API request.",
    )
    p.add_argument(
        "--judge-model",
        metavar="NAME",
        default="local-model",
        help="Model name sent in the judge API request.",
    )
    p.add_argument(
        "--judge-url",
        metavar="URL",
        default=None,
        help="Separate base URL for the judge model (defaults to --base-url).",
    )

    # --- Pipeline params ---
    p.add_argument(
        "--max-rounds",
        type=int,
        default=2,
        help="Maximum number of detection/refinement rounds.",
    )
    p.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="Start index for processing a subset of images.",
    )
    p.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="End index (exclusive) for processing a subset of images.",
    )

    # --- vLLM / model serving ---
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="vLLM model name or path.",
    )
    p.add_argument(
        "--download_dir",
        type=str,
        default=None,
        help="vLLM model download directory.",
    )
    p.add_argument(
        "--limit_mm_per_prompt",
        type=str,
        default=None,
        help="vLLM limit multimodal items per prompt.",
    )
    p.add_argument(
        "--chat_template",
        type=str,
        default=None,
        help="vLLM chat template.",
    )
    p.add_argument(
        "--extra_args",
        action="append",
        default=None,
        help="Extra arguments to pass to vLLM server.",
    )

    # --- Image encoding ---
    p.add_argument(
        "--image_min_tokens",
        type=int,
        default=1024,
        help="Minimum number of tokens to use for image encoding.",
    )
    p.add_argument(
        "--image_max_tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens to use for image encoding.",
    )
    p.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Height of the input image for the model.",
    )
    p.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Width of the input image for the model.",
    )

    # --- Labeling / output ---
    p.add_argument(
        "--init_class_map",
        action="store_true",
        help="Initialize the class map from the YAML file.",
    )
    p.add_argument(
        "--inplace_saving",
        action="store_true",
        help="Save inplace the relabeled annotations in the original label folder "
        "instead of a separate output folder.",
    )

    args = p.parse_args()

    if args.start_index is not None and args.start_index < 0:
        p.error("--start_index must be >= 0")
    if (
        args.end_index is not None
        and args.start_index is not None
        and args.end_index <= args.start_index
    ):
        p.error("--end_index must be greater than --start_index")

    return args


if __name__ == "__main__":
    # parse args
    args = build_parser()

    if args.task == "auto_label":
        aa_main(args)
    elif args.task == "free_detection":
        free_detection(args)

    else:
        raise ValueError("Invalid run type")
