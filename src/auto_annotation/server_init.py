"""Server / client lifecycle for the auto-annotation pipeline.

Spins up a local llama.cpp or vLLM server, waits for it to become healthy,
then returns an OpenAI-compatible client pointed at it. For the
``external`` server type it just returns a client pointing at a
user-supplied base_url.
"""

import json
import time
import urllib.error
import urllib.request

from openai import OpenAI

from servers import servers_factory
from .logging_utils import logger

# Back-compat alias: the original single-file module imported `factory` from
# `servers`, but `servers/__init__.py` only exports `servers_factory`. Keep
# both names so any external caller that still uses `factory` keeps working.
factory = servers_factory


def wait_for_server_health(port, timeout=1200, poll_interval=2.0):
    """
    Poll the llama.cpp /health endpoint until it returns a status of 200 ('ok')
    or we hit the timeout threshold.
    """
    url = f"http://localhost:{port}/health"
    start_time = time.time()
    logger.info(f"Probing server health at {url} (max timeout: {timeout}s)...")

    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if response.status == 200:
                    try:
                        data = json.loads(response.read().decode())
                        if data.get("status") == "ok":
                            logger.info(
                                "Server is healthy, model is loaded, and ready to process requests."
                            )
                            return True
                    except Exception:
                        logger.info("Server responded with 200. Proceeding.")
                        return True
        except urllib.error.HTTPError as e:
            # HTTP 503 means the server is online but still loading the model weights
            if e.code == 503:
                try:
                    err_data = json.loads(e.read().decode())
                    msg = err_data.get("error", {}).get("message", "Loading model")
                    logger.info(
                        f"Server is online but model is still loading: '{msg}'..."
                    )
                except Exception:
                    logger.info("Server is online but still loading the model (503)...")
            else:
                logger.warning(f"Server returned unexpected HTTP status: {e.code}")
        except Exception as e:
            # Quietly wait if connection is refused (server process hasn't fully bound to the port yet)
            logger.debug(f"Could not connect to server port yet: {e}")

        time.sleep(poll_interval)

    logger.error(
        f"Timed out waiting for server to become healthy after {timeout} seconds."
    )
    return False


def init_server(args):
    if args.server_type == "llama_cpp":
        manager = servers_factory[args.server_type](
            model=args.model,
            host="localhost",
            port=args.port,
            ctx_size=args.ctx_size,
            parallel_slots=args.parallel_slots,
            n_threads=-1,
            gpu_layers=-1,
            tensor_split="1,1",
            main_gpu=0,
            temp=0.1,
            top_p=0.85,
            top_k=24,
            spec_type="draft-mtp" if args.use_mtp else "none",
            spec_draft_n_max=4 if args.use_mtp else 0,
            fa="auto",
            enable_thinking=args.enable_thinking,
            batch_size=1024,
            ubatch_size=1024,
            kv_cache_type="q4_0",
            image_min_tokens=args.image_min_tokens,
            image_max_tokens=args.image_max_tokens,
        )
        manager.start_llama_server()
    elif args.server_type == "vllm":
        manager = servers_factory[args.server_type](
            model=args.model,
            host="localhost",
            port=args.port,
            max_model_len=getattr(args, "max_model_len", 20000),
            gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.90),
            tensor_parallel_size=getattr(args, "tensor_parallel_size", 1),
            pipeline_parallel_size=getattr(args, "pipeline_parallel_size", 1),
            dtype=getattr(args, "dtype", "auto"),
            quantization=getattr(args, "quantization", None),
            kv_cache_dtype=getattr(args, "kv_cache_dtype", "auto"),
            max_num_seqs=getattr(args, "max_num_seqs", 16),
            enforce_eager=getattr(args, "enforce_eager", False),
            enable_chunked_prefill=getattr(args, "enable_chunked_prefill", True),
            enable_prefix_caching=getattr(args, "enable_prefix_caching", True),
            speculative_model=getattr(args, "speculative_model", None),
            tokenizer_mode=getattr(args, "tokenizer_mode", "auto"),
            trust_remote_code=getattr(args, "trust_remote_code", True),
            download_dir=getattr(args, "download_dir", None),
        )
        # vLLM manager surfaces its own readiness; still poll HTTP health before
        # handing the client back so callers have a single wait-for-ready contract.
        manager.start_vllm_server()
    else:
        # Should never happen (argparse choices restrict this), but keep a
        # defensive guard so a future server type fails loudly here rather
        # than producing an UnboundLocalError on `manager` below.
        raise ValueError(f"Unsupported server_type for local serving: {args.server_type!r}")

    # Active HTTP polling replaces the static event wait logic
    server_ready = wait_for_server_health(args.port, timeout=1200, poll_interval=20.0)
    if not server_ready:
        logger.warning(
            "Proceeding, but server health checks did not pass successfully."
        )

    return manager


def init_vllm_server(args):
    vllm_manager = servers_factory["vllm"](
        model=args.model,
        host="localhost",
        port=args.port,
        max_model_len=getattr(args, "max_model_len", 20000),
        gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.90),
        tensor_parallel_size=getattr(args, "tensor_parallel_size", 1),
        pipeline_parallel_size=getattr(args, "pipeline_parallel_size", 1),
        dtype=getattr(args, "dtype", "auto"),
        quantization=getattr(args, "quantization", None),
        kv_cache_dtype=getattr(args, "kv_cache_dtype", "auto"),
        max_num_seqs=getattr(args, "max_num_seqs", 16),
        enforce_eager=getattr(args, "enforce_eager", False),
        enable_chunked_prefill=getattr(args, "enable_chunked_prefill", True),
        enable_prefix_caching=getattr(args, "enable_prefix_caching", True),
        speculative_model=getattr(args, "speculative_model", None),
        num_speculative_tokens=getattr(args, "num_speculative_tokens", None),
        trust_remote_code=getattr(args, "trust_remote_code", True),
        limit_mm_per_prompt=getattr(args, "limit_mm_per_prompt", None),
        chat_template=getattr(args, "chat_template", None),
        extra_args=getattr(args, "extra_args", None),
    )
    vllm_manager.start_vllm_server()
    return vllm_manager


def build_client(args):
    """Build the OpenAI-compatible client.

    --server_type external -> talk to an external OpenAI-compatible API
    (base_url / api_key). Anything else -> spin up a local llama.cpp or
    vLLM server and point the client at localhost.

    Returns (client, manager) where `manager` is the local server manager
    (so the caller can shut it down in a finally block) or None for the
    external case.
    """
    if args.server_type != "external":
        manager = init_server(args)
        client = OpenAI(
            base_url=f"http://localhost:{args.port}/v1", api_key="not-needed"
        )
        return client, manager

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    return client, None
