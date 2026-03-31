#!/bin/bash
#SBATCH -p lightwork
#SBATCH -q class
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-01:00:00
#SBATCH --mem=8G
#SBATCH -o outputs/out/clone_repo_%j.out
#SBATCH -e outputs/err/clone_repo_%j.err

git clone https://github.com/andytsai104/pedestrian_RL