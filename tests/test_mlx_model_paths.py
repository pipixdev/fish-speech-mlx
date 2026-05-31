import json
import tempfile
import unittest
from pathlib import Path

from fish_speech.inference_engine.mlx_defaults import (
    DEFAULT_MLX_MODEL_PATH,
    DEFAULT_MLX_STT_MODEL_PATH,
    LOCAL_FISH_BF16_DIR_NAME,
    LOCAL_WHISPER_FP16_DIR_NAME,
)
from fish_speech.inference_engine.mlx_engine import resolve_mlx_model_path


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _create_fish_bf16_model(root: Path) -> Path:
    model_dir = root / LOCAL_FISH_BF16_DIR_NAME
    _write_json(model_dir / "config.json", {"model_type": "fish_qwen3_omni"})
    _touch(model_dir / "codec.safetensors")
    _touch(model_dir / "model.safetensors.index.json")
    return model_dir


def _create_fish_8bit_model(root: Path) -> Path:
    model_dir = root / "fish-audio-s2-pro-8bit"
    _write_json(
        model_dir / "config.json",
        {
            "model_type": "fish_qwen3_omni",
            "quantization": {"bits": 8, "group_size": 64, "mode": "affine"},
        },
    )
    _touch(model_dir / "codec.safetensors")
    _touch(model_dir / "model.safetensors")
    _touch(model_dir / "model.safetensors.index.json")
    return model_dir


def _create_whisper_fp16_model(root: Path) -> Path:
    model_dir = root / LOCAL_WHISPER_FP16_DIR_NAME
    _write_json(
        model_dir / "config.json",
        {"model_type": "whisper", "torch_dtype": "float16"},
    )
    _touch(model_dir / "model.safetensors.index.json")
    _touch(model_dir / "preprocessor_config.json")
    return model_dir


class MLXModelPathResolutionTest(unittest.TestCase):
    def test_resolves_default_tts_repo_to_local_bf16_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tts_dir = _create_fish_bf16_model(root)

            resolved = resolve_mlx_model_path(
                DEFAULT_MLX_MODEL_PATH, "tts", models_root=root
            )

            self.assertEqual(resolved, str(tts_dir))

    def test_resolves_default_stt_repo_to_local_whisper_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stt_dir = _create_whisper_fp16_model(root)

            resolved = resolve_mlx_model_path(
                DEFAULT_MLX_STT_MODEL_PATH, "stt", models_root=root
            )

            self.assertEqual(resolved, str(stt_dir))

    def test_resolves_models_root_to_requested_child(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tts_dir = _create_fish_bf16_model(root)
            stt_dir = _create_whisper_fp16_model(root)

            self.assertEqual(resolve_mlx_model_path(root, "tts"), str(tts_dir))
            self.assertEqual(resolve_mlx_model_path(root, "stt"), str(stt_dir))

    def test_keeps_actual_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tts_dir = _create_fish_bf16_model(root)

            self.assertEqual(resolve_mlx_model_path(tts_dir, "tts"), str(tts_dir))

    def test_keeps_actual_quantized_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tts_dir = _create_fish_8bit_model(root)

            self.assertEqual(resolve_mlx_model_path(tts_dir, "tts"), str(tts_dir))

    def test_keeps_unknown_repo_id(self) -> None:
        self.assertEqual(
            resolve_mlx_model_path("mlx-community/other-model", "tts"),
            "mlx-community/other-model",
        )


if __name__ == "__main__":
    unittest.main()
