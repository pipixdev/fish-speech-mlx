import os
from pathlib import Path

DEFAULT_MLX_MODEL_PATH = "mlx-community/fish-audio-s2-pro-bf16"
DEFAULT_MLX_STT_MODEL_PATH = "mlx-community/whisper-large-v3-turbo-asr-fp16"
LOCAL_FISH_BF16_DIR_NAME = "fish-audio-s2-pro-bf16-audio-s2-pro-bf16"
LOCAL_WHISPER_FP16_DIR_NAME = "whisper-large-v3-turbo-asr-fp16"


def default_mlx_models_dir() -> Path:
    return Path(
        os.environ.get("FISH_MLX_MODELS_DIR", "/Users/pipix/Documents/Projects/models")
    )
