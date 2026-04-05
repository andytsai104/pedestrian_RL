#!/bin/bash
#SBATCH -p lightwork
#SBATCH -q class
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-00:05:00
#SBATCH --mem=2G
#SBATCH -o outputs/out/clone_repo_%j.out
#SBATCH -e outputs/err/clone_repo_%j.err

set -e

REPO_URL="https://github.com/andytsai104/pedestrian_RL.git"
REPO_DIR="pedestrian_RL"
BRANCH="main"

if [ ! -d "$REPO_DIR/.git" ]; then
    echo "Repo not found. Cloning..."
    git clone -b "$BRANCH" "$REPO_URL"
else
    echo "Repo exists. Force updating..."
    cd "$REPO_DIR"
    git fetch origin
    git reset --hard "origin/$BRANCH"
    git clean -fd
fi