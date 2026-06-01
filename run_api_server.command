#!/bin/zsh

set -u

PROJECT_DIR="/Users/pipix/Documents/Projects/fish-speech"
MODEL_ROOT="/Users/pipix/Documents/Projects/models"
MODEL_8BIT_PATH="$MODEL_ROOT/fish-audio-s2-pro-8bit"
MODEL_BF16_PATH="$MODEL_ROOT/fish-audio-s2-pro-bf16-audio-s2-pro-bf16"
MODEL_PATH=""
LISTEN_ADDR="0.0.0.0:8080"

export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

pause_before_exit() {
  echo
  echo "Press Enter to close this window."
  read -r _
}

choose_model_path() {
  local default_choice=""
  local selection=""

  if [[ -n "${FISH_MLX_MODEL_PATH:-}" ]]; then
    MODEL_PATH="$FISH_MLX_MODEL_PATH"
    return 0
  fi

  if [[ -d "$MODEL_8BIT_PATH" ]]; then
    default_choice="1"
  elif [[ -d "$MODEL_BF16_PATH" ]]; then
    default_choice="2"
  else
    default_choice="1"
  fi

  echo "Select MLX model:"
  echo "  1) 8bit  - $MODEL_8BIT_PATH"
  echo "  2) bf16  - $MODEL_BF16_PATH"
  echo

  while true; do
    printf "Choose model [1/2] (default: %s): " "$default_choice"
    read -r selection
    selection="${selection:-$default_choice}"

    case "$selection" in
      1)
        MODEL_PATH="$MODEL_8BIT_PATH"
        ;;
      2)
        MODEL_PATH="$MODEL_BF16_PATH"
        ;;
      *)
        echo "Invalid selection. Enter 1 or 2."
        echo
        continue
        ;;
    esac

    if [[ -d "$MODEL_PATH" ]]; then
      return 0
    fi

    echo "Model directory does not exist: $MODEL_PATH"
    echo
  done
}

if ! cd "$PROJECT_DIR"; then
  echo "Could not open project directory: $PROJECT_DIR"
  pause_before_exit
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Could not find 'uv'."
  echo "Install uv or make sure it is available in one of these paths:"
  echo "  /opt/homebrew/bin"
  echo "  /usr/local/bin"
  echo "  $HOME/.local/bin"
  echo "  $HOME/.cargo/bin"
  pause_before_exit
  exit 1
fi

choose_model_path

echo "Starting Fish Speech API Server"
echo "Project: $PROJECT_DIR"
echo "Model:   $MODEL_PATH"
echo "Listen:  $LISTEN_ADDR"
echo

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model directory does not exist: $MODEL_PATH"
  pause_before_exit
  exit 1
fi

uv run python tools/api_server.py \
  --backend mlx \
  --listen "$LISTEN_ADDR" \
  --mlx-model-path "$MODEL_PATH"

status=$?
echo
echo "Fish Speech API Server stopped with exit code $status."
pause_before_exit
exit "$status"
