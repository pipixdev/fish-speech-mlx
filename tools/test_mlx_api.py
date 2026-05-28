import argparse
import base64
import json
import math
import struct
import sys
import time
import wave
from pathlib import Path
from urllib import error, request


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REFERENCE_AUDIO = PROJECT_ROOT / "ref.mp3"
DEFAULT_REFERENCE_TEXT = PROJECT_ROOT / "ref_text.txt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke test the Fish Speech MLX API with reference audio."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8080",
        help="API base URL.",
    )
    parser.add_argument(
        "--text",
        default="你好，这是 Fish Speech MLX API 的参考音频测试。",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--reference-text",
        default=None,
        help="Transcript of the reference audio. Defaults to the contents of ref_text.txt in the project root.",
    )
    parser.add_argument(
        "--reference-audio",
        default=str(DEFAULT_REFERENCE_AUDIO),
        help="Path to a WAV/MP3/FLAC/OGG reference audio file. Defaults to ref.mp3 in the project root.",
    )
    parser.add_argument(
        "--reference-text-file",
        default=str(DEFAULT_REFERENCE_TEXT),
        help="Path to a UTF-8 text file containing the reference transcript. Defaults to ref_text.txt in the project root.",
    )
    parser.add_argument(
        "--output",
        default="test_output_mlx.wav",
        help="Path to save the synthesized audio.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Bearer token if the server was started with --api-key.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate used when generating fallback reference audio.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.2,
        help="Duration in seconds for generated fallback reference audio.",
    )
    return parser


def generate_reference_wav(path: Path, sample_rate: int, duration: float) -> None:
    frame_count = max(1, int(sample_rate * duration))
    amplitude = 0.18
    frequency = 220.0

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        frames = bytearray()
        for index in range(frame_count):
            sample = amplitude * math.sin(2.0 * math.pi * frequency * index / sample_rate)
            frames.extend(struct.pack("<h", int(sample * 32767)))
        wav_file.writeframes(frames)


def read_reference_audio(args: argparse.Namespace) -> tuple[bytes, str]:
    ref_path = Path(args.reference_audio).expanduser().resolve()
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference audio file not found: {ref_path}")
    return ref_path.read_bytes(), str(ref_path)


def read_reference_text(args: argparse.Namespace) -> str:
    if args.reference_text is not None:
        return args.reference_text

    ref_text_path = Path(args.reference_text_file).expanduser().resolve()
    if not ref_text_path.exists():
        raise FileNotFoundError(f"Reference text file not found: {ref_text_path}")
    return ref_text_path.read_text(encoding="utf-8").strip()


def build_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def get_json(url: str, headers: dict[str, str], timeout: float) -> dict:
    req = request.Request(url, headers=headers, method="GET")
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict, headers: dict[str, str], timeout: float) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method="POST")
    with request.urlopen(req, timeout=timeout) as response:
        return response.read()


def validate_wav_bytes(audio_bytes: bytes) -> None:
    if len(audio_bytes) < 44:
        raise ValueError("Response is too small to be a valid WAV file")
    if audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
        raise ValueError("Response does not look like a WAV file")


def main() -> int:
    args = build_parser().parse_args()
    base_url = args.base_url.rstrip("/")
    headers = build_headers(args.api_key)

    try:
        health = get_json(f"{base_url}/v1/health", headers, args.timeout)
        if health.get("status") != "ok":
            raise RuntimeError(f"Unexpected health response: {health}")

        audio_bytes, audio_source = read_reference_audio(args)
        reference_text = read_reference_text(args)
        if not reference_text:
            raise ValueError("Reference text is empty")
        payload = {
            "text": args.text,
            "format": "wav",
            "references": [
                {
                    "audio": base64.b64encode(audio_bytes).decode("ascii"),
                    "text": reference_text,
                }
            ],
        }

        started_at = time.perf_counter()
        response_audio = post_json(f"{base_url}/v1/tts", payload, headers, args.timeout)
        elapsed = time.perf_counter() - started_at
        validate_wav_bytes(response_audio)

        output_path = Path(args.output).expanduser().resolve()
        output_path.write_bytes(response_audio)

        print("Health check passed")
        print(f"Reference audio source: {audio_source}")
        print(f"Synthesis completed in {elapsed:.2f}s")
        print(f"Saved output to: {output_path}")
        return 0
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP error {exc.code}: {body}", file=sys.stderr)
        return 1
    except error.URLError as exc:
        print(f"Failed to reach server: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Test failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())