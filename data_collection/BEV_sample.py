"""
WORKFLOW: GROUND-TRUTH BIRD'S-EYE VIEW (BEV) FEATURE EXTRACTION
--------------------------------------------------------------
This process bypasses raw RGB "visual" cameras to provide the model with 
perfect spatial data (Semantic Masks) centered on a Target Actor.

1. ✅ACTOR CENTERED COORDINATE SYSTEM:
   - Identify the 'Hero' (Pedestrian or Vehicle).
   - Define a Top-Down 'Canvas' (e.g., 320x320 pixels) where the Hero 
     is fixed at the center.

2. ✅QUERY SIMULATOR STATE:
   - Use the CARLA Python API to retrieve the global (x, y) coordinates 
     of all relevant actors (Vehicles, Walkers, Traffic Lights).
   - Fetch the static Map geometry (Roads, Lanes, Sidewalks).

3. COORDINATE TRANSFORMATION & PROJECTION: -> Align everything to the Hero's perspective
   - Convert World Coordinates (Meters) -> Relative Coordinates (Meters from Hero).
   - Apply Rotation: Rotate the world around the Hero so that the Hero's 
     forward direction is always 'Up' (0°).
   - Scale to Pixels: Convert Relative Meters to Pixel indices based on 
     'pixels_per_meter' (Scale).

4. ✅SEMANTIC LAYER RENDERING (The "Dictionary"):
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
import cv2
import time

class BEVSample:
    '''
    Handles the "How to see" for one pedestrian.
    '''
    def __init__(self, actor, bev_wrapper):
        self.wrapper = bev_wrapper
        self.actor = actor
        self.wrapper.hero_actor = actor
        self.feature_tensor = None

    def get_BEV(self):
        # Focus the BEV engine on one pedestrian
        self.wrapper.hero_actor = self.actor
        
        # Extract the dictionary of layers
        layers = self.wrapper.get_bev_data()
        
        # 3. Stack them into a (320, 320, N) tensor
        # We use np.transpose to move channels to the front (C, H, W) 
        # if using PyTorch, or keep at end for TensorFlow.
        self.feature_tensor = np.stack([
            layers['lane'],
            layers['sidewalk'], 
            layers['vehicle'], 
            layers['pedestrian']
        ], axis=-1)
        
        return self.feature_tensor
    
    def get_velocity(self):
        vx, vy = self.actor.get_velocity().x, self.actor.get_velocity().y
        return np.array([vx, vy])

    def CNN_Encoder(self, model):
        '''
        Passes the tensor through your CNN (e.g., ResNet or MobileNet)
        to get a 1D Feature Vector (e.g., 512 dimensions).
        '''
        if self.feature_tensor is None:
            self.get_BEV()
        return model(self.feature_tensor)
    
    def visulize_bev(self):
        '''
        Get the output from self.get_bev_data() and visualize it as a single RGB image for debugging.
        lane: dark gray -> (40, 40, 40)
        sidewalk: gray -> (120, 120, 120)
        vehicle: red  -> (0, 0, 255)
        pedestrian: blue -> (255, 0, 0)

        HINT: OpenCV uses BRG format
        '''
        width = self.wrapper.width
        height = self.wrapper.height
        layers = self.wrapper.get_bev_data()
        self.wrapper.show_target_pedestrian()
        image = np.zeros((width, height, 3), dtype=np.uint8)
        image[layers['lane'] > 0] = (40, 40, 40)
        image[layers['sidewalk'] > 0] = (120, 120, 120)
        image[layers['vehicle'] > 0] = (0, 0, 255)
        image[layers['pedestrian'] > 0] = (255, 0, 0)
        center = (width // 2, height // 2)
        cv2.circle(image, center, 2, (0, 255, 255), -1)
        
        return image
    
class BEVWrapper:
    '''
    Tool box for generating BEV feature tensors from the CARLA world.
    '''
    config = dict(
        size=[320, 320],
        BEV_range = 16,                                             # get BEV in a 16mx16m area around the pedestrian
        pixels_per_meter= 20,                                       # resollution for BEV extractor: pixels/m
    )
    def __init__(self, cfg, world):
        self.image = None
        self.world = world
        self.hero_actor = None
        self.actor_list = None
        if cfg is not None:
            self.config = cfg
        else:
            self.config = BEVWrapper.config
        self.width = self.config['size'][0]
        self.height = self.config['size'][1]
        self.bev_range = self.config['BEV_range']                   # meters
        self.pixel_per_meter = self.config['pixels_per_meter']      # pixels/m


    def get_bev_data(self):
        '''
        1. Query the world for all relevant actors and map data around hero_ped.
        2. Transform their world coordinates to the hero-centered, rotated, pixel-based coordinate system.
        3. Render each type of actor/lane into its own binary mask layer.
        4. Return a dictionary of these layers, which can be stacked into a tensor.
        '''
        self.actor_list = self.world.get_actors()
        lane_canvas, sidewalks_canvas = self.draw_road_layers()
        return {
            'lane': lane_canvas,
            'sidewalk': sidewalks_canvas,
            'vehicle': self.draw_actor_layers('vehicle'),
            'pedestrian': self.draw_actor_layers('walker'),
            }
    
    def world_to_pixel(self, target_location):
        '''
        Convert world coordinates to pixels on canvas
        '''
        hero_transform = self.hero_actor.get_transform()
        hero_location = hero_transform.location
        hero_yaw = hero_transform.rotation.yaw

        # Distance between hero_ped and target_location
        dx = target_location.x - hero_location.x
        dy = target_location.y - hero_location.y

        # Make hero pedestrian always facing north (upwards)
        rad = np.radians(-hero_yaw)
        rx = - dx * np.cos(rad) + dy * np.sin(rad)
        ry = dx * np.sin(rad) + dy * np.cos(rad)

        px = int(self.width // 2 + rx * self.pixel_per_meter)
        py = int(self.height // 2 - ry * self.pixel_per_meter)

        return px, py
    
    def draw_actor_layers(self, actor_type):
        canvas = np.zeros((self.width, self.height), dtype=np.uint8)     # empty canvas
        actors = self.actor_list.filter(actor_type + ".*")
        
        for actor in actors:
            if actor.id == self.hero_actor.id: 
                continue
            px, py = self.world_to_pixel(actor.get_location())
            if 0 <= px < self.width and 0 <= py < self.height:
                cv2.circle(canvas, (px, py), radius=3, color=255, thickness=-1)

        return canvas
    
    def draw_road_layers(self):
        lane_canvas = np.zeros((self.width, self.height), dtype=np.uint8)           # empty canvas
        sidewalks_canvas = np.zeros((self.width, self.height), dtype=np.uint8)      # empty canvas

        # Define the range and step sizes for querying the map
        search_range = self.bev_range / 2
        step_size = 0.1        # sample every 0.25m -> adjust for higher resolution (but might require more computation)

        # Fill up the narrow between step sieze and pixel siezs to avoid holes on canvas
        brush_size = int(step_size * self.pixel_per_meter) + 1


        hero_transform = self.hero_actor.get_transform()
        # Loop through search area and determine lane type 
        for x in np.arange(-search_range, search_range, step_size):
            for y in np.arange(-search_range, search_range, step_size):
                # Get the world location of the target point relative to the hero pedestrian
                target_vector = carla.Vector3D(x, y, 0)
                world_location = hero_transform.transform(target_vector)
                wp = self.world.get_map().get_waypoint(world_location, project_to_road=False, lane_type=carla.LaneType.Any)

                if wp:
                    # Convert world location to pixel on the canvas
                    px, py = self.world_to_pixel(world_location)
                    lane_type = wp.lane_type

                    # Determine which canvas to draw
                    if 0 <= px < self.width and 0 <= py < self.height:
                        if lane_type == carla.LaneType.Driving:
                            cv2.rectangle(lane_canvas,
                                            (px - brush_size//2, py - brush_size//2),
                                            (px + brush_size//2, py + brush_size//2),
                                            color=255,
                                            thickness=-1)
                        elif (lane_type == carla.LaneType.Sidewalk) or (lane_type == carla.LaneType.Shoulder):
                            cv2.rectangle(sidewalks_canvas,
                                            (px - brush_size//2, py - brush_size//2),
                                            (px + brush_size//2, py + brush_size//2),
                                            color=255,
                                            thickness=-1)

        return lane_canvas, sidewalks_canvas
    
    def show_target_pedestrian(self):
        # Visualize target pedestrian in CARLA
        if self.hero_actor:
            location = self.hero_actor.get_location()
            location.z = 20
            self.world.debug.draw_point(location, size=0.3, color=carla.Color(0, 0, 255), life_time=2.0)
    





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

def find_pedestrian(world, test_ped=None):
    test_ped = None
    print("Waiting for pedestrians spawning...")

    while test_ped is None:
        # world.tick() to ensure pedestrians are spawned
        world.tick() 
        ped_list = world.get_actors().filter('walker.*')
        
        if len(ped_list) > 0:
            test_ped = ped_list[0]
            print(f"Find target pedestrian!! ID: {test_ped.id}")
        else:
            time.sleep(0.5)

    return test_ped

if __name__ == "__main__":
    '''Test the BEV_sample class functionality'''

    # Set up CARLA world and spawn actors as in intersection_sim.py
    client = carla.Client('localhost', 2000)
    world = client.get_world()

    while True:
        test_ped = find_pedestrian(world)
        bev_wrapper = BEVWrapper(cfg=None, world=world)
        bev_sample = BEVSample(actor=test_ped, bev_wrapper=bev_wrapper)
        
        while test_ped.is_alive:
            world.tick()
            image = bev_sample.visulize_bev()
            cv2.imshow("BEV Debug Tool", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        