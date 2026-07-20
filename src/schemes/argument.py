from __future__ import annotations

import argparse
from typing import List, Optional, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PipelineConfig(BaseModel):
    """Validated configuration for the unified relabeler / detector pipeline."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # --- Logging -----------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_file: Optional[str] = None

    # --- Input images ------------------------------------------------------
    # Detection mode: explicit image list (may be empty when relabeling from folders).
    images: List[str] = Field(default_factory=list)
    train_image: Optional[str] = None
    train_label: Optional[str] = None
    yaml_path: Optional[str] = None
    image_extensions: str = ".jpg,.jpeg,.png"

    # --- Categories --------------------------------------------------------
    categories: str = "person, car, bicycle, dog, cat"
    definitions: str = ""
    init_class_map: bool = False
    conf_threshold: Literal[1, 2, 3, 4, 5] = 2

    # --- Output ------------------------------------------------------------
    output_folder: str = "./detection_results"
    inplace_saving: bool = False
    no_plot: bool = False

    # --- Sampling / slicing ------------------------------------------------
    num_samples: Optional[int] = None
    shuffle: bool = False
    seed: int = 42
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    batch_size: int = 0
    dry_run: bool = False

    # --- Resume ------------------------------------------------------------
    resume: bool = False
    auto_resume: bool = True

    # --- Server / model ----------------------------------------------------
    model: str = "local-model"
    detector_model: str = "local-model"
    judge_model: str = "local-model"
    judge_url: Optional[str] = None
    api_key: str = "not-needed"
    base_url: str = "http://localhost:8080/v1"
    server_type: Literal["llama_cpp", "vllm", "external"] = "llama_cpp"
    max_workers: int = 1

    # llama.cpp-specific
    enable_thinking: bool = False
    use_mtp: bool = True
    ctx_size: int = 20000
    port: int = 8080
    parallel_slots: int = 1

    # --- Detection pipeline tuning ----------------------------------------
    max_rounds: int = 2
    score_threshold: int = 8
    detector_temperature: float = 0.9
    detector_top_p: float = 0.95
    judge_temperature: float = 0.2
    detector_max_tokens: int = 4096
    judge_max_tokens: int = 1024
    api_retries: int = 3

    # --- vLLM configuration ------------------------------------------------
    max_model_len: int = 20000
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    dtype: str = "auto"
    quantization: Optional[str] = None
    kv_cache_dtype: str = "auto"
    max_num_seqs: int = 1
    enforce_eager: bool = False
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True
    speculative_model: Optional[str] = None
    num_speculative_tokens: Optional[int] = None
    trust_remote_code: bool = True
    download_dir: Optional[str] = None
    limit_mm_per_prompt: Optional[str] = None
    chat_template: Optional[str] = None

    # --- VLM image encoding -----------------------------------------------
    image_min_tokens: int = 1024
    image_max_tokens: int = 4096
    height: int = 1024
    width: int = 1024

    # --- Preprocessing -----------------------------------------------------
    prep_enabled: bool = False
    prep_short_edge: int = 1024
    prep_pad_square: bool = False
    prep_contrast_method: Literal["none", "clahe", "autocontrast"] = "none"
    prep_gamma: float = 1.0
    prep_denoise_method: Literal["none", "bilateral", "nlm"] = "none"
    prep_sharpen: bool = False
    prep_white_balance: bool = False
    prep_grid_style: Literal["standard", "transparent", "fine", "none"] = "standard"
    prep_som_enabled: bool = False
    prep_tiling_enabled: bool = False
    prep_tile_size: int = 512
    prep_tile_overlap: float = 0.2
    prep_crop_verify_enabled: bool = False
    prep_crop_padding: float = 0.15

    # Custom grid overlays
    prep_grid_step: int = 100
    prep_grid_line_width: int = 1
    prep_grid_font_size: int = 0
    prep_grid_line_color: str = "red"
    prep_grid_text_color: str = "white"
    prep_grid_backing_color: str = "black"

    # VLM processor pixels
    prep_send_pixel_bounds: bool = False
    prep_min_pixels: int = 200_704
    prep_max_pixels: int = 4_194_304

    serving_extra: Optional[List[str]] = None

    # ------------------------------------------------------------------ validators
    @field_validator("prep_tile_overlap")
    @classmethod
    def _check_overlap(cls, v: float) -> float:
        if not 0.0 <= v <= 0.5:
            raise ValueError("prep_tile_overlap must be between 0.0 and 0.5")
        return v

    @field_validator("gpu_memory_utilization")
    @classmethod
    def _check_gpu_mem(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        return v

    @model_validator(mode="after")
    def _check_indices(self) -> "PipelineConfig":
        if self.start_index is not None and self.start_index < 0:
            raise ValueError("--start_index must be >= 0")
        if (
            self.end_index is not None
            and self.start_index is not None
            and self.end_index <= self.start_index
        ):
            raise ValueError("--end_index must be greater than --start_index")
        return self

    @model_validator(mode="after")
    def _check_input_source(self) -> "PipelineConfig":
        """Either an explicit image list OR a train_image folder must be supplied."""
        if not self.images and not self.train_image:
            raise ValueError(
                "Provide either --image (one or more) or --train_image (folder)."
            )
        return self

    @property
    def vllm(self):

        args = self.serving_extra
        # arguments for vllm
        args["--tensor_split"] = self.tensor_parallel_size
        args["--pipeline_parallel_size"] = self.pipeline_parallel_size
        args["--dtype"] = self.dtype
        args["--kv_cache_dtype"] = self.kv_cache_dtype
        args["--max_num_seqs"] = self.max_num_seqs
        args["--enforce_eager"] = self.enforce_eager
        args["--enable_chunked_prefill"] = self.enable_chunked_prefill
        args["--enable_prefix_caching"] = self.enable_prefix_caching
        args["--speculative_model"] = self.speculative_model
        args["--num_speculative_tokens"] = self.num_speculative_tokens
        args["--trust_remote_code"] = self.trust_remote_code
        args["--download_dir"] = self.download_dir
        args["--limit_mm_per_prompt"] = self.limit_mm_per_prompt
        args["--chat_template"] = self.chat_template
        args["--quantization"] = self.quantization
        args["--model"] = self.model
        return args

    @property
    def llama_cpp(self):
        args = self.serving_extra

        # 1. Base Model & Context Setup
        args["-m"] = self.model
        args["--ctx-size"] = self.ctx_size if hasattr(self, "ctx_size") else 2048
        args["--port"] = self.port if hasattr(self, "port") else 8080

        # 2. KV Cache Data Type Mapping
        # vLLM choice mapping (e.g., 'fp16', 'bf16', 'fp8', 'q8_0', 'q4_0')
        if hasattr(self, "kv_cache_dtype") and self.kv_cache_dtype:
            # Standardize vLLM 'fp8' / 'auto' naming to typical llama.cpp cache types
            dtype_map = {
                "fp16": "f16",
                "bf16": "f16",
                "fp8": "q8_0",  # 'q8_0' is standard 8-bit cache quantization in llama.cpp
                "fp8_e5m2": "q8_0",
                "fp8_e4m3": "q8_0",
            }
            # Fallback directly to the string if user directly passed llama.cpp formats (e.g. 'q4_0')
            target_dtype = dtype_map.get(self.kv_cache_dtype, self.kv_cache_dtype)

            args["--cache-type-k"] = target_dtype  # Configures Key cache type
            args["--cache-type-v"] = target_dtype  # Configures Value cache type

        # 3. Concurrency & Queue Capacity
        # vLLM max_num_seqs determines simultaneously active request slots
        args["--parallel"] = (
            self.parallel_slots
            if (hasattr(self, "parallel_slots") and self.parallel_slots)
            else self.max_num_seqs
        )

        # 4. Multi-GPU Splitting & Execution Controls
        args["--gpu-layers"] = 999  # Mandate all layer processing offloads to GPU
        if hasattr(self, "tensor_parallel_size") and self.tensor_parallel_size > 1:
            args["--split-mode"] = "layer"

        if hasattr(self, "enforce_eager") and self.enforce_eager:
            # Flash Attention optimization toggle can be turned off if eager is enforced
            args["--flash-attn"] = False
        else:
            args["--flash-attn"] = True

        # 5. Caching & Batch Strategies
        if hasattr(self, "enable_prefix_caching") and self.enable_prefix_caching:
            args["--cont-batching"] = True  # Continuous batching handles reuse prompts

        if hasattr(self, "enable_chunked_prefill") and self.enable_chunked_prefill:
            args["--batch-size"] = 512  # Restricts chunk limits per step optimization

        # 6. Speculative Decoding Configurations
        if hasattr(self, "speculative_model") and self.speculative_model:
            args["--model-draft"] = self.speculative_model  # Secondary draft model path
            if hasattr(self, "num_speculative_tokens") and self.num_speculative_tokens:
                args["--n-predict"] = self.num_speculative_tokens

        # 7. Modern Toggle Overrides (Reasoning & MTP)
        if hasattr(self, "enable_thinking"):
            args["--reasoning"] = "on" if self.enable_thinking else "off"

        if hasattr(self, "use_mtp") and self.use_mtp:
            args["--spec-type"] = "draft-mtp"

        return args
