#!/bin/bash
#SBATCH -p general
#SBATCH -q class
#SBATCH -n 1
#SBATCH -c 12
#SBATCH -t 0-02:00:00
#SBATCH --mem=20G
#SBATCH -o outputs/out/pedestrian_RL/bc_training_%j.out
#SBATCH -e outputs/err/pedestrian_RL/bc_training_%j.err

set -e

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/ctsai67/envs/carla_env

# Go to project
cd ~/pedestrian_RL

# Debug info
echo "Python: $(which python)"
python --version
python -c "import sys; print(sys.executable)"

# Run training
python -m pedestrian_rl.training.bc_training