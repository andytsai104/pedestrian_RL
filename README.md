# pedestrian RL


## Run the Simulation
- (Optional) Activate virtual env using conda
```text
conda activate env
```
-  Run CARLA simulator
```text
cd ~/CARLA_0.9.16/
./CarlaUE4.sh
```
- Run the intersection BEV simulation
```text
python -m intersection_sim.intersection_sim
```

- Visulize BEV sampling 
```text
python -m data_collection.BEV_sample
```

- for my setup to visuallize BEV data
```text
./show_bev_data.sh
```
exit session
```text
Ctrl B
:kill-session
```