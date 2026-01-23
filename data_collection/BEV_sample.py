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


class BEV_sample:
    '''
    Sample one single pedestrian's BEV
    IN: world, ped_id
    OUT: image feature tensor
    '''
    def __init__(self, world, ped_id):
        self.world = world
        self.ped_id = ped_id
        self.image = None
    
    def get_BEV(self):
        pass
  
    def CNN_Encoder(self):
        '''
        Get the input image tensor and encode it into feature vector 
        '''
        pass
    
class BEV_wrapper:
    
    '''
    Wrap up the state-acrion pairs for the pedestrians.
    State: {BEV tensor (320, 320, N), relative direction(x, y), current speed(vx, vy), Distance to vehicles}
    Actions: {vx, vy, time_stamp(t, frame)}
    '''
    def __init__(self):
        pass
    

class 