import numpy as np
import carla
from ..utils.config_loader import load_config

# Function for sampling data
def data_sampling():
    '''
    connect to simulation and sample the data for every existing pedestrian.
    '''
    config = load_config("data_sampling_config.json")
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    while True:
        '''
        world.tick() -> make sure the simulation restarted successully
        1. Get all pedestrian's list
        2. Check if the pedestrian is alive: actors.is_alive
        3. Store peds' info into PedestrianStateAction()
        4. Export to a data file (file type TBD)
        '''
        world.tick()

if __name__ == "__main__":

    data_sampling()