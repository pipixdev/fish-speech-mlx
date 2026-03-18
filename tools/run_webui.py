import os
from argparse import ArgumentParser
from pathlib import Path

import pyrootutils
from loguru import logger

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from tools.webui import build_app
from tools.webui.inference import get_inference_wrapper

# Make einx happy
os.environ["EINX_FILTER_TRACEBACK"] = "false"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--llama-checkpoint-path",
        type=Path,
        default="checkpoints/s2-pro",
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=Path,
        default="checkpoints/s2-pro/codec.pth",
    )
    parser.add_argument("--decoder-config-name", type=str, default="modded_dac_vq")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-gradio-length", type=int, default=0)
    parser.add_argument("--theme", type=str, default="light")

    # MLX backend options
    parser.add_argument(
        "--backend",
        type=str,
        choices=["torch", "mlx"],
        default="torch",
        help=(
            "Inference backend to use. "
            "'torch' uses the original PyTorch stack (LLaMA + DAC). "
            "'mlx' uses mlx_audio (Apple Silicon only)."
        ),
    )
    parser.add_argument(
        "--mlx-model-path",
        type=str,
        default="mlx-community/fish-audio-s2-pro-bf16",
        help="HuggingFace repo id or local path for the MLX model.",
    )
    parser.add_argument(
        "--mlx-lang-code",
        type=str,
        default="auto",
        help="BCP-47 language code passed to mlx_audio (e.g. 'ja', 'zh', 'en', 'auto').",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ------------------------------------------------------------------ #
    # MLX branch                                                          #
    # ------------------------------------------------------------------ #
    if args.backend == "mlx":
        logger.info("[MLX] Using mlx_audio backend – skipping PyTorch models.")
        from fish_speech.inference_engine.mlx_engine import MLXTTSInferenceEngine

        inference_engine = MLXTTSInferenceEngine(
            model_path=args.mlx_model_path,
            lang_code=args.mlx_lang_code,
        )
        logger.info("[MLX] Engine ready, launching the web UI…")
        inference_fct = get_inference_wrapper(inference_engine)
        app = build_app(inference_fct, args.theme)
        app.launch()

    # ------------------------------------------------------------------ #
    # Original PyTorch branch (unchanged)                                 #
    # ------------------------------------------------------------------ #
    else:
        import torch

        from fish_speech.inference_engine import TTSInferenceEngine
        from fish_speech.models.dac.inference import load_model as load_decoder_model
        from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
        from fish_speech.utils.schema import ServeTTSRequest

        args.precision = torch.half if args.half else torch.bfloat16

        # Check if MPS or CUDA is available
        if torch.backends.mps.is_available():
            args.device = "mps"
            logger.info("mps is available, running on mps.")
        elif torch.xpu.is_available():
            args.device = "xpu"
            logger.info("XPU is available, running on XPU.")
        elif not torch.cuda.is_available():
            logger.info("CUDA is not available, running on CPU.")
            args.device = "cpu"

        logger.info("Loading Llama model...")
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=args.llama_checkpoint_path,
            device=args.device,
            precision=args.precision,
            compile=args.compile,
        )

        logger.info("Loading VQ-GAN model...")
        decoder_model = load_decoder_model(
            config_name=args.decoder_config_name,
            checkpoint_path=args.decoder_checkpoint_path,
            device=args.device,
        )

        logger.info("Decoder model loaded, warming up...")

        # Create the inference engine
        inference_engine = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            compile=args.compile,
            precision=args.precision,
        )

        # Dry run to check if the model is loaded correctly and avoid the first-time latency
        list(
            inference_engine.inference(
                ServeTTSRequest(
                    text="Hello world.",
                    references=[],
                    reference_id=None,
                    max_new_tokens=1024,
                    chunk_length=200,
                    top_p=0.7,
                    repetition_penalty=1.5,
                    temperature=0.7,
                    format="wav",
                )
            )
        )

        logger.info("Warming up done, launching the web UI...")

        # Get the inference function with the immutable arguments
        inference_fct = get_inference_wrapper(inference_engine)

        app = build_app(inference_fct, args.theme)
        app.launch()
