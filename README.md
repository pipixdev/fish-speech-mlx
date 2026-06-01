# Fish Speech API Server

This repository has been trimmed for one use case: running the Fish Speech API
server with `mlx_audio` on Apple Silicon and calling it from the local Speech
client at `/Users/pipix/Documents/Projects/Speech`.

## Supported Models

The local launcher supports these two Fish MLX model directories:

- `/Users/pipix/Documents/Projects/models/fish-audio-s2-pro-8bit`
- `/Users/pipix/Documents/Projects/models/fish-audio-s2-pro-bf16-audio-s2-pro-bf16`

The verified project pin is `mlx-audio==0.4.2`. Do not bump to
`mlx-audio==0.4.3` without revalidation; it loads the models, but the bf16
model regresses to obviously bad/noisy speech output on this setup.

## Install Dependencies

```bash
uv sync --extra mlx
```

## Start The API Server

Double-click:

```text
run_api_server.command
```

The launcher prompts for the 8bit or bf16 model and starts the server on
`0.0.0.0:8080`.

To skip the prompt:

```bash
FISH_MLX_MODEL_PATH=/Users/pipix/Documents/Projects/models/fish-audio-s2-pro-8bit \
  ./run_api_server.command
```

Direct command:

```bash
uv run python tools/api_server.py \
  --backend mlx \
  --listen 0.0.0.0:8080 \
  --mlx-model-path /Users/pipix/Documents/Projects/models/fish-audio-s2-pro-8bit
```

API available locally at `http://127.0.0.1:8080` and on your LAN at
`http://<this-machine-lan-ip>:8080`.

Keep the default `--workers 1`. Each worker loads an independent MLX model and
Metal runtime.

## API Surface

Kept endpoints:

- `GET /v1/health`
- `POST /v1/health`
- `POST /v1/tts`
- `POST /v1/references/add`
- `GET /v1/references/list`
- `DELETE /v1/references/delete`
- `POST /v1/references/update`

Removed from this trimmed project:

- WebUI and frontend assets
- Docker and Nvidia/CUDA runtime paths
- PyTorch LLaMA/DAC backend
- VQGAN encode/decode endpoints
- Training, fine-tuning, datasets, docs site, and i18n tooling

## API Parameter Behavior

Parameters forwarded to `mlx_audio`:

- `text`
- `references` and `reference_id`
- `max_new_tokens`
- `top_p`
- `repetition_penalty`
- `temperature`
- `streaming`
- `format` for the final HTTP response encoding

Parameters still accepted for client compatibility but currently no-op:

- `seed`
- `normalize`
- `latency`
- `use_memory_cache`
- `chunk_length`

The engine keeps generation internal to WAV. The API server re-encodes the
final response into the requested output format.

## Tests

Run the unit tests:

```bash
uv run python -m unittest tests.test_mlx_model_paths tests.test_mlx_request_mapping
```

Run all tests, including the real API integration test when the machine has
`mlx` and `mlx_audio` installed:

```bash
uv run python -m unittest discover -s tests
```

Integration test configuration:

| Variable | Default | Description |
| --- | --- | --- |
| `FISH_MLX_MODELS_DIR` | `/Users/pipix/Documents/Projects/models` | Local flat model root |
| `FISH_MLX_TEST_MODEL` | `mlx-community/fish-audio-s2-pro-bf16` | TTS model repo id, model root, or explicit local Fish path |
| `FISH_MLX_TEST_STT_MODEL` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | STT model repo id, model root, or explicit local Whisper path |
| `FISH_MLX_TEST_STARTUP_TIMEOUT` | `600` | Seconds to wait for server startup |
| `FISH_MLX_TEST_TTS_TIMEOUT` | `300` | Seconds to wait for TTS generation |
