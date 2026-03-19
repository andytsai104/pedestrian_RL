#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG_FILE="$SCRIPT_DIR/user_config.sh"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
fi

CARLA_PATH="${CARLA_PATH:-$HOME/CARLA_0.9.16}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-D2RL}"
SESSION_NAME="${SESSION_NAME:-ped_bev_debug}"
CARLA_ARGS="${CARLA_ARGS:-$CARLA_NVIDIA_ARG}"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' already exists."
    echo "Attach with: tmux attach -t $SESSION_NAME"
    exit 1
fi

tmux new-session -d -s "$SESSION_NAME"

# Pane 1: CARLA
tmux send-keys -t "$SESSION_NAME" \
  "cd \"$CARLA_PATH\" && ./CarlaUE4.sh $CARLA_ARGS" C-m

echo "Waiting for CARLA to initialize..."
while ! pgrep -f "CarlaUE4" > /dev/null; do
    sleep 2
done
sleep 10

# Pane 2: intersection sim
tmux split-window -h -t "$SESSION_NAME"
tmux send-keys -t "$SESSION_NAME" \
  "source \"$CONDA_SH\" && conda activate \"$CONDA_ENV\" && cd \"$REPO_ROOT\" && python -m pedestrian_rl.simulation.intersection_sim" C-m
sleep 5

# Pane 3: BEV display
tmux split-window -v -t "$SESSION_NAME:0.1"
tmux send-keys -t "$SESSION_NAME" \
  "source \"$CONDA_SH\" && conda activate \"$CONDA_ENV\" && cd \"$REPO_ROOT\" && python -m pedestrian_rl.data_collection.bev.bev_sample" C-m

tmux attach-session -t "$SESSION_NAME"