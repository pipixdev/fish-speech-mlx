# Fish Speech — MLX Backend (Apple Silicon)

> **Note:** Docker support for the MLX backend has **not been verified**. Running MLX on Docker is not expected to work, as it requires direct access to Apple Silicon hardware and the Metal framework, which are not available inside containers.

An inference backend powered by `mlx_audio`

## Install Dependencies

```bash
uv sync --extra mlx
```

The verified project pin is `mlx-audio==0.4.2`. On this machine it supports
both local Fish model directories:

- `/Users/pipix/Documents/Projects/models/fish-audio-s2-pro-8bit`
- `/Users/pipix/Documents/Projects/models/fish-audio-s2-pro-bf16-audio-s2-pro-bf16`

Do not bump to `mlx-audio==0.4.3` for this repo without revalidation. It loads
the models, but the bf16 model regresses to obviously bad/noisy speech output.

## Start the Service

Quick-start launchers on this machine:

- `run_api_server_mlx.command` prompts you to choose the 8bit or bf16 model at startup
- Set `FISH_MLX_MODEL_PATH` to skip the prompt and force an exact model directory

### WebUI (Gradio, recommended)

```bash
uv run python tools/run_webui.py \
  --backend mlx \
  --mlx-model-path /Users/pipix/Documents/Projects/models/fish-audio-s2-pro-8bit
```

Open your browser at http://127.0.0.1:7860

### API Server

```bash
uv run python tools/api_server.py \
  --backend mlx \
  --listen 0.0.0.0:8080 \
  --mlx-model-path /Users/pipix/Documents/Projects/models/fish-audio-s2-pro-8bit
```

API available locally at http://127.0.0.1:8080 and on your LAN at
`http://<this-machine-lan-ip>:8080`.

When both the bf16 and 8bit Fish models exist under `FISH_MLX_MODELS_DIR`, pass
the exact Fish model directory you want. The bare models root still resolves the
default bf16 shortcut directory.

For long-running API service, keep the default `--workers 1`. The MLX backend
uses one shared model instance per worker, serializes generation inside each
process, clears MLX/Metal runtime caches after each request, and removes its
temporary work directory on server shutdown.

## Tests

The MLX integration tests live under `tests/integration/`. They start a real
API server in a subprocess, wait for `/v1/health`, call `/v1/tts` with a
temporary reference WAV, and verify that the response is a valid WAV file. This
is intentionally a real model test, not a mocked unit test.

### Run the MLX API server test

```bash
uv run python -m unittest tests.integration.test_mlx_api_server
```

The test requires macOS with `mlx` and `mlx_audio` installed. On other platforms,
or when the MLX dependencies are missing, it is skipped.

What the test covers:

- starts `tools/api_server.py --backend mlx --workers 1`
- waits for `/v1/health` to return `{"status": "ok"}`
- generates a short temporary WAV as reference audio
- sends a real `/v1/tts` JSON request with base64 reference audio
- checks that the synthesized response is a WAV payload
- terminates the server and removes temporary files after the test

### Test configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `FISH_MLX_MODELS_DIR` | `/Users/pipix/Documents/Projects/models` | Local flat model root containing the bf16 TTS and Whisper fp16 ASR directories |
| `FISH_MLX_TEST_MODEL` | `mlx-community/fish-audio-s2-pro-bf16` | HuggingFace repo id, model root, or explicit local Fish TTS path used by the test server |
| `FISH_MLX_TEST_STT_MODEL` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | HuggingFace repo id, model root, or local Whisper fp16 ASR path used when reference text is omitted |
| `FISH_MLX_TEST_STARTUP_TIMEOUT` | `600` | Seconds to wait for server startup and model loading |
| `FISH_MLX_TEST_TTS_TIMEOUT` | `300` | Seconds to wait for the real TTS request |

Example using a local model path:

```bash
FISH_MLX_TEST_MODEL=/Users/pipix/Documents/Projects/models/fish-audio-s2-pro-8bit \
  uv run python -m unittest tests.integration.test_mlx_api_server
```

If startup fails or the TTS request fails, the test includes the tail of the
temporary API server log in the failure message.

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mlx-model-path` | [`mlx-community/fish-audio-s2-pro-bf16`](https://huggingface.co/mlx-community/fish-audio-s2-pro-bf16) | HuggingFace repo id, model root, or explicit local Fish TTS path |
| `--mlx-stt-model-path` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | HuggingFace repo id, model root, or local Whisper fp16 ASR path |
| `--mlx-lang-code` | `auto` | Language code, e.g. `ja` / `zh` / `en`; `auto` for automatic detection |

When the default repo ids are used, the backend first looks for these local
directories under `FISH_MLX_MODELS_DIR`:

- `fish-audio-s2-pro-bf16-audio-s2-pro-bf16`
- `fish-audio-s2-pro-8bit` (pass this exact directory when you want the quantized model)
- `whisper-large-v3-turbo-asr-fp16`

## MLX API Parameter Behavior

The MLX API mode accepts the normal `ServeTTSRequest` payload, but only the
subset that `mlx_audio` supports is forwarded into the MLX generator.

Parameters that currently take effect in MLX API mode:

- `text`
- `references` and `reference_id`
- `max_new_tokens`
- `top_p`
- `repetition_penalty`
- `temperature`
- `streaming`
- `format` for the final HTTP response encoding

Parameters that are still accepted for compatibility but are currently no-ops
in MLX API mode:

- `seed`
- `normalize`
- `latency`
- `use_memory_cache`
- `chunk_length`

Notes:

- The MLX engine keeps its internal generation format as WAV and the API server
  re-encodes the final response into the requested output format.
- `streaming=true` still uses the existing Fish Speech response wrapper. The
  current MLX path does not yet expose true progressive chunk generation.
