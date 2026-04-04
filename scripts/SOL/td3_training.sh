#!/bin/bash
#SBATCH -G a100:1
#SBATCH -p general
#SBATCH -q class
#SBATCH -n 1
#SBATCH -c 12
#SBATCH -t 0-08:00:00
#SBATCH --mem=32G
#SBATCH -o outputs/out/pedestrian_RL/td3_training_%j.out
#SBATCH -e outputs/err/pedestrian_RL/td3_training_%j.err

set -e

# ----- user settings -----
REPO_URL="https://github.com/andytsai104/pedestrian_RL.git"
REPO_DIR="$HOME/pedestrian_RL"
BRANCH="main"
CARLA_PATH="${CARLA_PATH:-$HOME/CARLA_0.9.16}"
CARLA_ARGS="${CARLA_ARGS:---RenderOffScreen -preferNvidia}"
CARLA_PORT="${CARLA_PORT:-2000}"
CARLA_GPU="${CARLA_GPU:-0}"

# change this if your SOL conda env is different
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/scratch/ctsai67/envs/carla_env}"
# -------------------------

mkdir -p outputs/out/pedestrian_RL
mkdir -p outputs/err/pedestrian_RL
mkdir -p "$HOME/.carla_logs"

# ----- clone / update repo -----
if [ ! -d "$REPO_DIR/.git" ]; then
    echo "Repo not found. Cloning..."
    git clone -b "$BRANCH" "$REPO_URL" "$REPO_DIR"
else
    echo "Repo exists. Force updating..."
    cd "$REPO_DIR"
    git fetch origin
    git reset --hard "origin/$BRANCH"
    git clean -fd
fi

# ----- activate env -----
source activate "$CONDA_ENV_PATH"

cd "$REPO_DIR"

# ----- debug info -----
echo "Node: $(hostname)"
echo "Python: $(which python)"
python --version
python -c "import sys; print(sys.executable)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# ----- start CARLA in background -----
export CUDA_VISIBLE_DEVICES="$CARLA_GPU"
export SDL_VIDEODRIVER=offscreen

echo "Starting CARLA server..."
"$CARLA_PATH/CarlaUE4.sh" $CARLA_ARGS -carla-port=$CARLA_PORT > "$HOME/.carla_logs/td3_carla_${SLURM_JOB_ID}.log" 2>&1 &
CARLA_PID=$!

echo "CARLA PID: $CARLA_PID"

cleanup() {
    echo "Stopping CARLA..."
    if ps -p $CARLA_PID > /dev/null 2>&1; then
        kill $CARLA_PID || true
        wait $CARLA_PID || true
    fi
}
trap cleanup EXIT

# wait for CARLA to initialize
sleep 15

if ! ps -p $CARLA_PID > /dev/null 2>&1; then
    echo "CARLA failed to start. Check log: $HOME/.carla_logs/td3_carla_${SLURM_JOB_ID}.log"
    exit 1
fi

# ----- run TD3 training -----
echo "Starting TD3 training..."
python -m pedestrian_rl.training.td3_training