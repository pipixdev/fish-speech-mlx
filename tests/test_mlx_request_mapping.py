import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from fish_speech.inference_engine.mlx_engine import MLXTTSInferenceEngine
from fish_speech.utils.schema import ServeTTSRequest


def _fake_mlx_audio_modules(generate_audio, load_model=None):
    if load_model is None:
        load_model = lambda _: object()

    mlx_audio_module = types.ModuleType("mlx_audio")
    tts_module = types.ModuleType("mlx_audio.tts")
    generate_module = types.ModuleType("mlx_audio.tts.generate")
    utils_module = types.ModuleType("mlx_audio.tts.utils")
    generate_module.generate_audio = generate_audio
    utils_module.load_model = load_model
    tts_module.generate = generate_module
    tts_module.utils = utils_module
    mlx_audio_module.tts = tts_module
    return {
        "mlx_audio": mlx_audio_module,
        "mlx_audio.tts": tts_module,
        "mlx_audio.tts.generate": generate_module,
        "mlx_audio.tts.utils": utils_module,
    }


class MLXRequestMappingTest(unittest.TestCase):
    def test_qwen_engine_uses_native_sample_rate_and_auto_language(self) -> None:
        qwen_model = types.SimpleNamespace(
            model_type="qwen3_tts",
            sample_rate=24000,
        )

        with (
            patch(
                "fish_speech.inference_engine.mlx_engine._check_mlx_audio",
                return_value=True,
            ),
            patch.dict(
                sys.modules,
                _fake_mlx_audio_modules(
                    lambda **_: None,
                    load_model=lambda _: qwen_model,
                ),
            ),
        ):
            engine = MLXTTSInferenceEngine(
                model_path="/models/Qwen3-TTS-12Hz-1.7B-Base-bf16",
                lang_code="auto",
                stt_model_path=None,
            )

        self.assertEqual(engine.sample_rate, 24000)
        self.assertEqual(engine.decoder_model.sample_rate, 24000)
        self.assertEqual(engine._generation_lang_code, "auto")

    def test_run_inference_forwards_supported_sampling_params(self) -> None:
        captured_kwargs: dict[str, object] = {}

        def fake_generate_audio(**kwargs):
            captured_kwargs.update(kwargs)
            output_path = Path(str(kwargs["output_path"]))
            output_path.mkdir(parents=True, exist_ok=True)
            with wave.open(str(output_path / "out_000.wav"), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(44100)
                wav_file.writeframes(b"\x00\x00" * 64)

        engine = MLXTTSInferenceEngine.__new__(MLXTTSInferenceEngine)
        engine._mlx_model = object()
        engine._stt_model_path = "/tmp/fake-whisper"
        engine.lang_code = "auto"
        engine._generation_lang_code = None
        engine.sample_rate = 44100
        engine._clear_runtime_cache = lambda: None

        request = ServeTTSRequest(
            text="Test MLX API request mapping.",
            references=[],
            format="mp3",
            seed=123,
            normalize=False,
            max_new_tokens=321,
            top_p=0.55,
            repetition_penalty=1.35,
            temperature=0.65,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            engine._tmp_dir = Path(temp_dir)
            engine._ensure_tmp_dir = lambda: Path(temp_dir)
            engine._resolve_reference = (
                lambda req: ("/tmp/reference.wav", "Reference transcript.", None)
            )

            with patch.dict(
                sys.modules,
                _fake_mlx_audio_modules(fake_generate_audio),
            ):
                result = engine._run_inference(request)

        self.assertEqual(result.code, "final")
        self.assertIsNone(result.error)
        self.assertEqual(result.audio[0], 44100)
        self.assertGreater(len(result.audio[1]), 0)

        self.assertEqual(captured_kwargs["model"], engine._mlx_model)
        self.assertEqual(captured_kwargs["text"], request.text)
        self.assertEqual(captured_kwargs["ref_audio"], "/tmp/reference.wav")
        self.assertEqual(captured_kwargs["ref_text"], "Reference transcript.")
        self.assertEqual(captured_kwargs["stt_model"], "/tmp/fake-whisper")
        self.assertEqual(captured_kwargs["max_tokens"], request.max_new_tokens)
        self.assertEqual(captured_kwargs["top_p"], request.top_p)
        self.assertEqual(
            captured_kwargs["repetition_penalty"], request.repetition_penalty
        )
        self.assertEqual(captured_kwargs["temperature"], request.temperature)
        self.assertEqual(captured_kwargs["audio_format"], "wav")
        self.assertIsNone(captured_kwargs["lang_code"])
        self.assertNotIn("seed", captured_kwargs)
        self.assertNotIn("normalize", captured_kwargs)


if __name__ == "__main__":
    unittest.main()
