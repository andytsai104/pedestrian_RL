import numpy as np
import carla

# Function for sampling data
def data_sampling(world):
    '''
    connect to simulation and sample the data for every existing pedestrian.
    '''
    
    while True:
        '''
        world.tick() -> make sure the simulation restarted successully
        1. Get all pedestrian's list
        2. Check if the pedestrian is alive: actors.is_alive
        3. Store peds' info into PedestrianStateAction()
        4. Export to a data file (file type TBD)
        '''