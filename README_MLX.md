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

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mlx-model-path` | [`mlx-community/fish-audio-s2-pro-bf16`](https://huggingface.co/mlx-community/fish-audio-s2-pro-bf16) | HuggingFace repo id or local path |
| `--mlx-lang-code` | `auto` | Language code, e.g. `ja` / `zh` / `en`; `auto` for automatic detection |