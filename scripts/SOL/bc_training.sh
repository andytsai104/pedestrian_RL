#!/bin/bash
#SBATCH -p general
#SBATCH -q class
#SBATCH -n 1
#SBATCH -c 12
#SBATCH -t 0-02:00:00
#SBATCH --mem=20G
#SBATCH -o outputs/out/pedestrian_RL/bc_training_%j.out
#SBATCH -e outputs/err/pedestrian_RL/bc_training_%j.err

# Go to pedestrian_RL project
cd pedestrian_RL/

# Run training
python -m pedestrian_rl.training.bc_training