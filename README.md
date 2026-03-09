# Pedestrian RL

This project implements a simulation framework in CARLA for studying pedestrian behavior and collecting training data for reinforcement learning and imitation learning.

The goal is to build a BEV-based pedestrian controller that can learn realistic and potentially aggressive crossing behavior in urban intersections.

The simulator generates traffic scenarios with vehicles and pedestrians, extracts bird's-eye-view (BEV) observations, and collects state–action data for training models such as Behavioral Cloning (BC) or Reinforcement Learning (RL).

--------------------------------------------------

## Features

- CARLA-based intersection simulation
- Pedestrian and vehicle scenario generation
- Bird's-Eye-View (BEV) observation extraction
- Dataset generation for learning pedestrian policies
- Modular project structure for simulation, data collection, and training

--------------------------------------------------

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
│   │   ├── data_sampling.py       (Collect state-action data)
│   │   └── state_action_pair.py
│   │
│   ├── simulation
│   │   ├── intersection_sim.py    (Main simulation entry)
│   │   ├── aggressive_vehicles.py
│   │   └── crossroad_pedestrians.py
│   │
│   ├── training
│   │   └── (model training scripts)
│   │
│   └── utils
│       ├── config_loader.py
│       └── intersection_sim_utils.py
│
├── scripts
│   └── show_bev_data.sh           (Visualize BEV data)
│
├── requirements.txt
|
└── README.md
```
--------------------------------------------------

## Requirements
```text
- Python 3.10+
- CARLA 0.9.16
- numpy
- opencv-python
```

Install dependencies using pip or conda.

Example:
```bash
pip install numpy opencv-python
```

### Install CARLA: 
Download CARLA from the [official website](https://carla.org/2025/09/16/release-0.9.16/) and [Official installation guide](https://carla.readthedocs.io/en/latest/start_quickstart/)

Example:
```bash
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.16.tar.gz
tar -xvzf CARLA_0.9.16.tar.gz
```
Then run the following to initialize CARLA simulation:
```bash
cd CARLA_0.9.16
./CarlaUE4.sh
```

### Create Python Environment (optional)
You can use either conda or venv:

#### Option 1: Conda
Create the virtual environment:
```bash
conda create -n pedestrian_rl python=3.10
conda activate pedestrian_rl
```
Install dependencies:
```bash
pip install -r requirements.txt
```

#### Option 2: Python venv
Create the virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```
Install dependencies:
```bash
pip install -r requirements.txt
```

--------------------------------------------------

## Run the Simulation

(Optional) activate your environment
```bash
conda activate env
```

Start CARLA simulator
```bash
cd ~/CARLA_0.9.16
./CarlaUE4.sh
```

Run the intersection simulation
```bash
python -m pedestrian_rl.simulation.intersection_sim
```
--------------------------------------------------

## Visualizing BEV Observations

Run the Simulation first then run BEV visualization module:
```bash
python -m pedestrian_rl.data_collection.bev.bev_sample
```
Or use the helper script:
```bash
./scripts/show_bev_data.sh
```
To exit a running session:

```bash
Ctrl + B
:kill-session
```
--------------------------------------------------

Configuration

Simulation parameters are stored in:
```bash
configs/sim_config.json
```
The configuration file controls:

- CARLA town
- simulation timestep
- intersection location
- vehicle and pedestrian spawning
- BEV resolution
- dataset settings

--------------------------------------------------

Dataset Generation

The simulator can collect data consisting of:

- BEV observations
- pedestrian states
- actions

These samples can be used for:

- Behavioral Cloning
- Reinforcement Learning
- Offline RL

Datasets are stored in the datasets/ folder.

--------------------------------------------------

Future Work

- Train pedestrian policies with RL (e.g., TD3)
- Improve pedestrian decision models
- Multi-agent pedestrian simulation
- Better traffic interaction modeling

--------------------------------------------------

License

This project is for research and educational purposes.