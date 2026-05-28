# Fish Speech — MLX Backend (Apple Silicon)

> **Note:** Docker support for the MLX backend has **not been verified**. Running MLX on Docker is not expected to work, as it requires direct access to Apple Silicon hardware and the Metal framework, which are not available inside containers.

An inference backend powered by `mlx_audio`

## Install Dependencies

```bash
uv sync --extra mlx
```

## Start the Service

### WebUI (Gradio, recommended)

```bash
uv run python tools/run_webui.py \
  --backend mlx \
  --mlx-model-path mlx-community/fish-audio-s2-pro-bf16
```

Open your browser at http://127.0.0.1:7860

### API Server

```bash
uv run python tools/api_server.py \
  --backend mlx \
  --mlx-model-path mlx-community/fish-audio-s2-pro-bf16
```

API available at http://127.0.0.1:8080

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
| `FISH_MLX_TEST_MODEL` | `mlx-community/fish-audio-s2-pro-bf16` | HuggingFace repo id or local path used by the test server |
| `FISH_MLX_TEST_STARTUP_TIMEOUT` | `600` | Seconds to wait for server startup and model loading |
| `FISH_MLX_TEST_TTS_TIMEOUT` | `300` | Seconds to wait for the real TTS request |

Example using a local model path:

```bash
FISH_MLX_TEST_MODEL=/path/to/mlx-model \
  uv run python -m unittest tests.integration.test_mlx_api_server
```

If startup fails or the TTS request fails, the test includes the tail of the
temporary API server log in the failure message.

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mlx-model-path` | [`mlx-community/fish-audio-s2-pro-bf16`](https://huggingface.co/mlx-community/fish-audio-s2-pro-bf16) | HuggingFace repo id or local path |
| `--mlx-lang-code` | `auto` | Language code, e.g. `ja` / `zh` / `en`; `auto` for automatic detection |
