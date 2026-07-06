# AGENTS.md — LLM Object Detection Testing Console

## Project Overview
Interactive test console for assessing Vision-Language Models (VLMs) on object detection tasks. Uses an iterative **Detector-Judge pipeline** where a detector proposes bounding boxes, a judge critiques them, and the loop repeats with structured feedback.

## Package Management
- **Tool**: `uv` (fast Python installer/resolver)
- **Python**: 3.12+ (see `.python-version`)
- **Install**: `uv sync` (after running `scripts/install_llama_cpp.sh` for llama.cpp)

## Entry Points
| Command | Module | Description |
|---------|--------|-------------|
| `uv run detection-gui` | `gui:main` | Launch Gradio web interface |
| `uv run detection-cli` | `cli:main` | Run pipeline from terminal |

## Key Directories
- `src/` — Main source code (package root via `tool.setuptools.package-dir`)
- `src/prompts/` — Markdown prompt templates for detector/judge agents
- `src/interface/` — Gradio console theme (CSS/JS)
- `scripts/` — Installation scripts (Linux-focused)

## Core Modules
- `detection_pipeline.py` — Core `ObjectDetectionPipeline` class, prompt loading, JSON parsing, retry logic, NMS, tiling, crop-verify
- `image_preprocessing.py` — Resolution tuning, CLAHE/autocontrast, gamma, bilateral/NLM denoise, unsharp mask, white balance, SoM proposals, grid drawing, tiling
- `llama_server_manager.py` — Local `llama-server` process management (start/stop/logs)
- `app.py` — Gradio web UI (server management, batch testing, live results, zip export)
- `cli.py` — CLI argument parsing, pipeline execution, per-image output

## Common Commands
```bash
# Install dependencies
./scripts/install_llama_cpp.sh  # Linux only; builds llama.cpp with CUDA
uv sync

# Run GUI (default: http://0.0.0.0:7860)
uv run detection-gui
uv run detection-gui --port 7861 --share

# Run CLI (single image)
uv run detection-cli -i image.jpg -c "person, car, dog"

# Run CLI (batch with preprocessing)
uv run detection-cli \
  -i img1.jpg -i img2.jpg \
  -c "crack, scratch, dent" \
  --prep-enabled --prep-contrast-method clahe \
  --prep-tiling-enabled --prep-tile-size 512 \
  --prep-grid-line-color blue --prep-grid-step 50 \
  -o ./results
```

## Important Conventions
- **Prompt templates**: Loaded from `src/prompts/*.md` with hardcoded fallbacks in `detection_pipeline.py`
- **Output structure**: Each image gets a subdir under `--output-dir` with `best_annotated.jpg`, `best_detections.json`, `history.json`
- **API compatibility**: Uses OpenAI Python SDK; works with any OpenAI-compatible endpoint (`llama-server`, vLLM, Ollama)
- **Extra body params**: `min_pixels`/`max_pixels` sent via `extra_body` for Qwen-VL/vLLM backends (controlled by `--prep-send-pixel-bounds`)
- **External API mode**: When enabled, only official OpenAI parameters are sent (no vLLM extensions, sampling params can be disabled)

## Development Notes
- No test suite currently exists
- No lint/typecheck configured in `pyproject.toml`
- Gradio UI loads custom CSS/JS from `src/interface/` at startup
- Matplotlib backend forced to 'Agg' in `detection_pipeline.py` (line 27)
- Logging uses standard `logging` module with `[LEVEL] message` format