#!/bin/bash

# Define your paths
CARLA_PATH="$HOME/CARLA_0.9.16"
PROJ_PATH="$HOME/ASU/BELIV/pedestrian_RL"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"

# 1. Start a new detached tmux session
tmux new-session -d -s pedestrian_rl

# --- STEP 1: START CARLA ---
tmux send-keys -t pedestrian_rl "cd $CARLA_PATH && ./CarlaUE4.sh" C-m

echo "Waiting for CARLA to initialize..."
# Wait until the CarlaUE4 process actually exists
while ! pgrep -f "CarlaUE4" > /dev/null; do
    sleep 2
done
sleep 15 # Extra buffer for the map to load

# --- STEP 2: START INTERSECTION SIM ---
tmux split-window -h -t pedestrian_rl
tmux send-keys -t pedestrian_rl "source $CONDA_SH && conda activate D2RL && cd $PROJ_PATH && python -m intersection_sim.intersection_sim" C-m

echo "Verifying Intersection Simulation..."
# Wait for the python process to be active
while ! pgrep -f "intersection_sim.intersection_sim" > /dev/null; do
    sleep 2
done
sleep 5 # Allow it to spawn pedestrians in the world

# --- STEP 3: START BEV SAMPLE ---
tmux split-window -v -t pedestrian_rl
tmux send-keys -t pedestrian_rl "source $CONDA_SH && conda activate D2RL && cd $PROJ_PATH && python -m data_collection.BEV_sample" C-m

# 5. Attach to session
tmux attach-session -t pedestrian_rl