"""
WORKFLOW: GROUND-TRUTH BIRD'S-EYE VIEW (BEV) FEATURE EXTRACTION
--------------------------------------------------------------
This process bypasses raw RGB "visual" cameras to provide the model with 
perfect spatial data (Semantic Masks) centered on a Target Actor.

1. ACTOR CENTERED COORDINATE SYSTEM:
   - Identify the 'Hero' (Pedestrian or Vehicle).
   - Define a Top-Down 'Canvas' (e.g., 320x320 pixels) where the Hero 
     is fixed at the center.

2. QUERY SIMULATOR STATE:
   - Use the CARLA Python API to retrieve the global (x, y) coordinates 
     of all relevant actors (Vehicles, Walkers, Traffic Lights).
   - Fetch the static Map geometry (Roads, Lanes, Sidewalks).

3. COORDINATE TRANSFORMATION & PROJECTION:
   - Convert World Coordinates (Meters) -> Relative Coordinates (Meters from Hero).
   - Apply Rotation: Rotate the world around the Hero so that the Hero's 
     forward direction is always 'Up' (0°).
   - Scale to Pixels: Convert Relative Meters to Pixel indices based on 
     'pixels_per_meter' (Scale).

4. SEMANTIC LAYER RENDERING (The "Dictionary"):
   - Instead of one RGB image, the process renders multiple Binary Masks:
     * 'road': Binary mask of drivable surfaces.
     * 'vehicle': Bounding boxes of all nearby cars.
     * 'pedestrian': Locations of all other pedestrians.
     * 'route': The desired path/waypoints for the agent.

5. FEATURE STACKING (CNN Input):
   - The individual layers (Dict Keys) are stacked along the channel axis.
   - Example Output Shape: (320, 320, N) where N is the number of layers.
   - This "Multi-Channel Tensor" is fed directly into the CNN Encoder.


"""

import numpy as np
import carla

class BEVSample:
    '''
    Handles the "How to see" for one pedestrian.
    '''
    def __init__(self, actor, bev_wrapper):
        self.wrapper = bev_wrapper
        self.actor = actor              # The actual carla.Walker object
        self.feature_tensor = None

    def get_BEV(self):
        # Focus the BEV engine on one pedestrian
        self.wrapper.world_module.hero_actor = self.actor
        
        # Extract the dictionary of layers
        layers = self.wrapper.get_bev_data()
        
        # 3. Stack them into a (320, 320, N) tensor
        # We use np.transpose to move channels to the front (C, H, W) 
        # if using PyTorch, or keep at end for TensorFlow.
        self.feature_tensor = np.stack([
            layers['road'], 
            layers['vehicle'], 
            layers['pedestrian']
        ], axis=-1)
        
        return self.feature_tensor

    def CNN_Encoder(self, model):
        '''
        Passes the tensor through your CNN (e.g., ResNet or MobileNet)
        to get a 1D Feature Vector (e.g., 512 dimensions).
        '''
        if self.feature_tensor is None:
            self.get_BEV()
        return model(self.feature_tensor)
    
class BEVWrapper:
    '''
    Tool box for generating BEV feature tensors from the CARLA world.
    '''
    config = dict(
        size=[320, 320],
        pixels_per_meter=5,
        pixels_ahead_vehicle=100,
    )
    def __init__(self, cfg, world):
        self.image = None
        self.world = world


    def get_bev_data(self):
        '''
        1. Create a blank canvas centered on the hero actor.
        2. Query the world for all relevant actors and map data.
        '''
    
    
    

class PedestrianStateAction:
    '''
    The "Memory" of your model: stores what the pedestrian saw vs. what they did.
    '''
    def __init__(self, actor_id):
        self.actor_id = actor_id
        self.state = {
            "bev": None,         # The (320, 320, N) tensor
            "velocity": None,    # (vx, vy): vx and vy can get heading and speed
            "dist_to_car": None, # (N, 1)
            "timestamp": None,   # (t, time_stamp)
        }
        self.action = {
            "target_vel": None,  # The vx, vy the model chose
            "timestamp": None,   # (t, time_stamp)
        }



if __name__ == "__main__":
    '''Test the BEV_sample class functionality'''
    