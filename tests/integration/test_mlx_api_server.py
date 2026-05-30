import base64
import contextlib
import json
import math
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path
from urllib import error, request

from fish_speech.inference_engine.mlx_defaults import (
    DEFAULT_MLX_MODEL_PATH,
    DEFAULT_MLX_STT_MODEL_PATH,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _find_free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _has_module(module_name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module_name) is not None


def _write_reference_wav(path: Path, sample_rate: int = 16000) -> None:
    duration = 1.0
    amplitude = 0.18
    frequency = 220.0
    frame_count = int(sample_rate * duration)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        frames = bytearray()
        for index in range(frame_count):
            sample = amplitude * math.sin(2.0 * math.pi * frequency * index / sample_rate)
            frames.extend(struct.pack("<h", int(sample * 32767)))
        wav_file.writeframes(frames)


def _request_json(url: str, timeout: float) -> dict:
    req = request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, payload: dict, timeout: float) -> bytes:
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        return response.read()


def _assert_wav(test_case: unittest.TestCase, audio: bytes) -> None:
    test_case.assertGreater(len(audio), 44)
    test_case.assertEqual(audio[:4], b"RIFF")
    test_case.assertEqual(audio[8:12], b"WAVE")


class MLXAPIServerIntegrationTest(unittest.TestCase):
    """Start the MLX API server and call the real model through HTTP."""

    server_process: subprocess.Popen | None = None
    temp_dir: tempfile.TemporaryDirectory | None = None
    log_file = None
    log_path: Path | None = None
    base_url: str

    @classmethod
    def setUpClass(cls) -> None:
        if sys.platform != "darwin":
            raise unittest.SkipTest("MLX integration tests require macOS.")
        if not _has_module("mlx") or not _has_module("mlx_audio"):
            raise unittest.SkipTest("MLX integration tests require mlx and mlx_audio.")

        cls.temp_dir = tempfile.TemporaryDirectory(prefix="fish_mlx_api_test_")
        temp_path = Path(cls.temp_dir.name)
        port = _find_free_port()
        cls.base_url = f"http://127.0.0.1:{port}"
        cls.log_path = temp_path / "api_server.log"
        cls.log_file = open(cls.log_path, "w", encoding="utf-8")

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("MPLCONFIGDIR", str(temp_path / "matplotlib"))

        model_path = os.environ.get("FISH_MLX_TEST_MODEL", DEFAULT_MLX_MODEL_PATH)
        stt_model_path = os.environ.get(
            "FISH_MLX_TEST_STT_MODEL", DEFAULT_MLX_STT_MODEL_PATH
        )
        command = [
            sys.executable,
            "tools/api_server.py",
            "--backend",
            "mlx",
            "--workers",
            "1",
            "--listen",
            f"127.0.0.1:{port}",
            "--mlx-model-path",
            model_path,
            "--mlx-stt-model-path",
            stt_model_path,
            "--max-text-length",
            "300",
        ]

        try:
            cls.server_process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=cls.log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            cls._wait_for_health()
        except Exception:
            cls._cleanup_server()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        cls._cleanup_server()

    @classmethod
    def _cleanup_server(cls) -> None:
        if cls.server_process is not None:
            cls.server_process.terminate()
            try:
                cls.server_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                cls.server_process.kill()
                cls.server_process.wait(timeout=30)
            cls.server_process = None
        if cls.log_file is not None:
            cls.log_file.close()
            cls.log_file = None
        if cls.temp_dir is not None:
            cls.temp_dir.cleanup()
            cls.temp_dir = None

    @classmethod
    def _wait_for_health(cls) -> None:
        startup_timeout = float(os.environ.get("FISH_MLX_TEST_STARTUP_TIMEOUT", "600"))
        deadline = time.monotonic() + startup_timeout
        last_error: Exception | None = None

        while time.monotonic() < deadline:
            if cls.server_process is not None and cls.server_process.poll() is not None:
                raise RuntimeError(
                    "MLX API server exited before becoming healthy.\n"
                    + cls._read_log_tail()
                )
            try:
                health = _request_json(f"{cls.base_url}/v1/health", timeout=2.0)
                if health.get("status") == "ok":
                    return
            except Exception as exc:
                last_error = exc
            time.sleep(1.0)

        raise TimeoutError(
            f"Timed out waiting for MLX API server health: {last_error}\n"
            + cls._read_log_tail()
        )

    @classmethod
    def _read_log_tail(cls, max_chars: int = 8000) -> str:
        if cls.log_path is None or not cls.log_path.exists():
            return ""
        return cls.log_path.read_text(encoding="utf-8", errors="replace")[-max_chars:]

    def test_health_endpoint(self) -> None:
        health = _request_json(f"{self.base_url}/v1/health", timeout=10.0)
        self.assertEqual(health, {"status": "ok"})

    def test_tts_generates_wav_with_real_mlx_model(self) -> None:
        assert self.temp_dir is not None
        reference_path = Path(self.temp_dir.name) / "reference.wav"
        _write_reference_wav(reference_path)

        payload = {
            "text": "你好，这是一次 MLX API 集成测试。",
            "format": "wav",
            "references": [
                {
                    "audio": base64.b64encode(reference_path.read_bytes()).decode("ascii"),
                    "text": "这是测试用的参考音频。",
                }
            ],
            "max_new_tokens": 128,
        }

        try:
            audio = _post_json(
                f"{self.base_url}/v1/tts",
                payload,
                timeout=float(os.environ.get("FISH_MLX_TEST_TTS_TIMEOUT", "300")),
            )
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            self.fail(f"TTS request failed with HTTP {exc.code}: {body}\n{self._read_log_tail()}")
        except Exception as exc:
            self.fail(f"TTS request failed: {exc}\n{self._read_log_tail()}")

        _assert_wav(self, audio)


if __name__ == "__main__":
    unittest.main()
