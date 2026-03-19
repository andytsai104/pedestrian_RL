#!/bin/bash
set -e

CONFIG_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/user_config.sh"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
fi

# ---- user-overridable settings ----
CARLA_PATH="${CARLA_PATH:-$HOME/CARLA_0.9.16}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-D2RL}"
SESSION_NAME="${SESSION_NAME:-ped_data_sampling}"
CARLA_ARGS="${CARLA_ARGS:-$CARLA_NO_RENDERING_ARG -$CARLA_NVIDIA_ARG}"
# -----------------------------------

# repo root = parent of /scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

tmux new-session -d -s "$SESSION_NAME"

# Pane 1: start CARLA
tmux send-keys -t "$SESSION_NAME" \
  "cd \"$CARLA_PATH\" && ./CarlaUE4.sh $CARLA_ARGS" C-m

echo "Waiting for CARLA to initialize..."
while ! pgrep -f "CarlaUE4" > /dev/null; do
    sleep 2
done
sleep 10

# Pane 2: start data sampling
tmux split-window -h -t "$SESSION_NAME"
tmux send-keys -t "$SESSION_NAME" \
  "source \"$CONDA_SH\" && conda activate \"$CONDA_ENV\" && cd \"$REPO_ROOT\" && python -m pedestrian_rl.simulation.data_sampling_sim" C-m

tmux attach-session -t "$SESSION_NAME"