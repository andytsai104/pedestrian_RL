# Pedestrian RL

This project implements a CARLA-based simulation framework for studying pedestrian behavior and collecting training data for reinforcement learning and imitation learning.

The goal is to build a BEV-based pedestrian controller that can learn realistic and potentially aggressive crossing behavior in urban intersections.

The simulator generates traffic scenarios with vehicles and pedestrians, extracts bird's-eye-view (BEV) observations, and collects state-action data for downstream policy learning. The current repository includes scripts for launching CARLA, running intersection scenarios, visualizing BEV observations, and sampling pedestrian datasets.

---

## Features

- CARLA-based intersection simulation
- Aggressive vehicle and crossroad pedestrian scenario generation
- Bird's-Eye-View (BEV) observation extraction
- Pedestrian state-action dataset generation
- Helper shell scripts for launching CARLA and Python modules in tmux
- Modular project structure for simulation, data collection, and training

---

## Project Structure

```text
pedestrian_RL
├── configs
│   └── sim_config.json            (Simulation configuration)
│
├── datasets                       (Generated datasets)
├── media                          (Visualization outputs)
│
├── pedestrian_rl
│   ├── data_collection
│   │   ├── bev                    (BEV feature extraction)
│   │   │   ├── bev_sample.py
│   │   │   └── cnn_encoder.py
│   │   │
│   │   ├── utils.py
│   │   └── state_action_pair.py
│   │
│   │
│   ├── models
│   │   ├── cnn_encoder.py
│   │   └── bc_policy.py
│   │
│   │
│   ├── simulation
│   │   ├── intersection_sim.py
│   │   └── data_sampling_sim.py
│   │
│   │
│   ├── training
│   │   ├── rl_training.py
│   │   └── bc_training.py
│   │
│   └── utils
│       ├── config_loader.py
│       └── sim_utils.py
│
├── scripts
│   ├── data_sampling.sh           (Launch CARLA + sample pedestrian data)
│   ├── show_bev_data.sh           (Launch CARLA + intersection sim + BEV visualization)
│   └── user_config.sh             (Local user-specific paths and launch settings)
│
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.10+
- CARLA 0.9.16
- tmux
- numpy
- opencv-python
- h5py

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Install CARLA

Download CARLA 0.9.16 from the official CARLA release page and extract it:

```bash
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.16.tar.gz
tar -xvzf CARLA_0.9.16.tar.gz
```

Test that CARLA launches correctly:

```bash
cd CARLA_0.9.16
./CarlaUE4.sh
```

---

## Create a Python Environment

You can use either conda or venv.

### Option 1: Conda

```bash
conda create -n pedestrian_rl python=3.10
conda activate pedestrian_rl
pip install -r requirements.txt
```

### Option 2: Python venv

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## User Configuration

The helper scripts read local settings from:

```bash
scripts/user_config.sh
```

Example:

```bash
#!/bin/bash

export CARLA_PATH="$HOME/CARLA_0.9.16"
export CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_ENV="D2RL"
export CARLA_NVIDIA_ARG="-preferNvidia"
export CARLA_NO_RENDERING_ARG="--RenderOffScreen"
```

You can edit this file to match your machine without modifying the main scripts.

### What each variable means

- `CARLA_PATH`: path to your CARLA installation
- `CONDA_SH`: path to `conda.sh` so the scripts can activate your environment
- `CONDA_ENV`: name of your conda environment
- `CARLA_NVIDIA_ARG`: optional CARLA launch argument for NVIDIA preference
- `CARLA_NO_RENDERING_ARG`: optional CARLA launch argument for offscreen rendering

---

## Run the Simulation Manually

Activate your environment first if needed:

```bash
conda activate pedestrian_rl
```

### Run the intersection scenario

Start CARLA in one terminal:

```bash
cd ~/CARLA_0.9.16
./CarlaUE4.sh
```

Then run the intersection simulation in another terminal from the repo root:

```bash
python -m pedestrian_rl.simulation.intersection_sim
```

This launches the intersection scenario with aggressive vehicles and pedestrians.

---

## Visualize BEV Observations

If you want to inspect the BEV representation without dataset sampling, you can run:

```bash
python -m pedestrian_rl.data_collection.bev.bev_sample
```

This expects the simulation world to already be running.

You can also use the helper script:

```bash
./scripts/show_bev_data.sh
```

This script:
1. launches CARLA,
2. starts the intersection simulation,
3. opens the BEV visualization module in another tmux pane.

---

## Dataset Generation

To generate pedestrian state-action samples, run:

```bash
python -m pedestrian_rl.simulation.data_sampling_sim
```

Or use the helper script:

```bash
./scripts/data_sampling.sh
```

This script:
1. launches CARLA,
2. activates the configured Python environment,
3. runs the data sampling pipeline.

The sampler stores:
- BEV observations
- pedestrian locations
- velocity
- speed
- motion heading
- goal location
- target speed
- target direction

Sampled data is saved to the dataset path defined in:

```bash
configs/sim_config.json
```

---

## Configuration

Simulation and dataset parameters are stored in:

```bash
configs/sim_config.json
```

This file controls:
- CARLA town
- simulation timestep
- intersection location
- vehicle spawn behavior
- pedestrian spawn behavior
- BEV size and range
- dataset save path and file name

---

## tmux Notes

Both helper scripts use `tmux`.

List running tmux sessions:

```bash
tmux ls
```

Attach to an existing session:

```bash
tmux attach -t ped_data_sampling
```

or

```bash
tmux attach -t ped_bev_debug
```

Kill a session:

```bash
tmux kill-session -t ped_data_sampling
```

To exit from inside tmux:

```bash
Ctrl + B
```

then type:

```bash
:kill-session
```

---

## Future Work

- Train pedestrian policies with RL (e.g., TD3)
- Improve pedestrian behavior modeling
- Multi-agent pedestrian simulation
- Add training and evaluation pipelines for BC / Offline RL

---

## License

This project is for research and educational purposes.
