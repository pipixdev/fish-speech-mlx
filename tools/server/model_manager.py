from loguru import logger

from fish_speech.inference_engine.mlx_defaults import (
    DEFAULT_MLX_MODEL_PATH,
    DEFAULT_MLX_STT_MODEL_PATH,
)


class ModelManager:
    def __init__(
        self,
        backend: str = "mlx",
        mlx_model_path: str = DEFAULT_MLX_MODEL_PATH,
        mlx_stt_model_path: str = DEFAULT_MLX_STT_MODEL_PATH,
        mlx_lang_code: str = "auto",
    ) -> None:
        self.backend = backend
        if backend != "mlx":
            raise ValueError("This project is MLX-only; use --backend mlx.")

        logger.info("[MLX] Loading mlx_audio backend.")
        from fish_speech.inference_engine.mlx_engine import MLXTTSInferenceEngine

        self.tts_inference_engine = MLXTTSInferenceEngine(
            model_path=mlx_model_path,
            lang_code=mlx_lang_code,
            stt_model_path=mlx_stt_model_path,
        )
        self.decoder_model = self.tts_inference_engine.decoder_model
        logger.info("[MLX] Engine ready.")
