import numpy as np
import carla
import cv2
import time
from ...utils.config_loader import load_config
import math


class BEVWrapper:
    '''
    Class for generating semantic bird's-eye-view layers from the CARLA world.

    This class provides the low-level tools for BEV generation. It keeps track of the
    hero actor, converts world coordinates into actor-centered pixel coordinates, draws
    static map layers such as lanes and sidewalks, and draws dynamic actor layers such
    as vehicles and pedestrians.

    Attributes:
        config: BEV-related configuration dictionary loaded from the config file.
        image: Placeholder for a rendered BEV image.
        world: CARLA world object.
        hero_actor: Current target actor that defines the center and orientation of the BEV.
        actor_list: Cached CARLA actor list used for drawing dynamic actors.
        width: Width of the BEV image in pixels.
        height: Height of the BEV image in pixels.
        bev_range: Physical range covered by the BEV in meters.
        pixel_per_meter: Conversion ratio from world meters to BEV pixels.

    Methods:
        get_bev_data():
            Generate and return a dictionary of BEV semantic layers.

        world_to_pixel(target_location):
            Convert a world location into BEV pixel coordinates relative to the hero actor.

        draw_actor_layers(actor_type):
            Draw a binary layer for dynamic actors of the given type, such as walkers or vehicles.

        draw_road_layers():
            Draw the static road-related BEV layers, including drivable lanes and sidewalks.

        show_target_pedestrian():
            Draw the current hero actor in the CARLA world for visualization and debugging.
    '''
    config = load_config("sim_config.json")["bev"]
    def __init__(self, cfg, world):
        self.image = None
        self.world = world
        self.hero_actor = None
        self.actor_list = None
        if cfg is not None:
            self.config = cfg
        else:
            self.config = BEVWrapper.config
        self.width = self.config["size"][0]
        self.height = self.config["size"][1]
        self.bev_range = self.config["range"]                       # meters
        self.pixel_per_meter = self.width // self.bev_range         # pixels/m
        self.hero_ped_size = self.config["hero_ped_size"]
        self.other_ped_size = self.config["other_ped_size"]
        self.step_size = self.config["step_size"]


    def get_bev_data(self):
        '''
        Return a dictionary of these layers, which can be stacked into a tensor later.
        '''
        self.actor_list = self.world.get_actors()
        lane_canvas, sidewalks_canvas, shoulder_canvas = self.draw_road_layers()
        return {
            "lane": lane_canvas,
            "sidewalk": sidewalks_canvas,
            "shoulder": shoulder_canvas,
            "vehicle": self.draw_actor_layers("vehicle"),
            "pedestrian": self.draw_actor_layers("walker"),
            }
    

    def world_to_pixel(self, target_location):
        '''
        Transform world location to image pixel
        '''
        hero_transform = self.hero_actor.get_transform()
        hero_location = hero_transform.location
        yaw = np.radians(hero_transform.rotation.yaw)

        dx = target_location.x - hero_location.x
        dy = target_location.y - hero_location.y

        # Hero local axes in world frame
        forward_x = np.cos(yaw)
        forward_y = np.sin(yaw)
        right_x = -np.sin(yaw)
        right_y =  np.cos(yaw)

        # Project world delta onto hero frame
        local_forward = dx * forward_x + dy * forward_y
        local_right   = dx * right_x   + dy * right_y

        # Map to image: right -> +x, forward -> up (-y)
        px = int(self.width  // 2 + local_right   * self.pixel_per_meter)
        py = int(self.height // 2 - local_forward * self.pixel_per_meter)

        return px, py
    
    def draw_actor_layers(self, actor_type):
        '''
        Draw layers for actors (walkers and vehicles)
        '''
        canvas = np.zeros((self.height, self.width), dtype=np.uint8)
        actors = self.actor_list.filter(actor_type + ".*")
        
        for actor in actors:
            extent = actor.bounding_box.extent
            
            # Pedestrians (Walkers) as Circles
            if actor_type == "walker":
                # hero pedestrin (larger mark and lighter feature)
                if actor.id == self.hero_actor.id:
                    color = 100
                    radius = self.hero_ped_size
                else:    
                    color = 255
                    radius = self.other_ped_size   
                px, py = self.world_to_pixel(actor.get_location())
                if 0 <= px < self.width and 0 <= py < self.height:
                    cv2.circle(canvas, (px, py), radius=radius, color=color, thickness=-1)
            
            # Vehicles as Rotated Rectangles
            elif actor_type == "vehicle":
                # Get the 4 corners of the vehicle in world coordinates
                transform = actor.get_transform()
                # Calculate the 4 corners relative to the vehicle center
                corners = [
                    carla.Location(x=-extent.x, y=-extent.y),
                    carla.Location(x=extent.x, y=-extent.y),
                    carla.Location(x=extent.x, y=extent.y),
                    carla.Location(x=-extent.x, y=extent.y)
                ]
                
                # Transform corners to world coordinates, then to pixels
                pixel_corners = []
                for corner in corners:
                    world_corner = transform.transform(corner)
                    px, py = self.world_to_pixel(world_corner)
                    pixel_corners.append([px, py])
                
                # Draw the rotated rectangle
                cv2.fillPoly(canvas, [np.array(pixel_corners, dtype=np.int32)], color=255)

        return canvas
    
    def draw_road_layers(self):
        lane_canvas = np.zeros((self.height, self.width), dtype=np.uint8)
        sidewalk_canvas = np.zeros((self.height, self.width), dtype=np.uint8)
        shoulder_canvas = np.zeros((self.height, self.width), dtype=np.uint8)

        search_range = self.bev_range / 2
        brush_size = int(self.step_size * self.pixel_per_meter) + 1

        hero_transform = self.hero_actor.get_transform()

        # Loop through the bev area
        for x in np.arange(-search_range, search_range, self.step_size):
            for y in np.arange(-search_range, search_range, self.step_size):
                target_vector = carla.Vector3D(x, y, 0)
                world_location = hero_transform.transform(target_vector)

                wp = self.world.get_map().get_waypoint(
                    world_location,
                    project_to_road=False,
                    lane_type=carla.LaneType.Any
                )

                if wp is None:
                    continue

                px, py = self.world_to_pixel(world_location)
                lane_type = wp.lane_type

                if not (0 <= px < self.width and 0 <= py < self.height):
                    continue
                
                # decide which layer to draw on
                if lane_type == carla.LaneType.Driving:
                    canvas = lane_canvas
                elif lane_type == carla.LaneType.Sidewalk:
                    canvas = sidewalk_canvas
                elif lane_type == carla.LaneType.Shoulder:
                    canvas = shoulder_canvas
                else:
                    continue

                cv2.rectangle(
                    canvas,
                    (px - brush_size // 2, py - brush_size // 2),
                    (px + brush_size // 2, py + brush_size // 2),
                    color=255,
                    thickness=-1
                )

        return lane_canvas, sidewalk_canvas, shoulder_canvas
    
    def show_target_pedestrian(self):
        '''
        Visualize target pedestrian in CARLA
        '''
        if self.hero_actor:
            location = self.hero_actor.get_location()
            location.z = 1
            self.world.debug.draw_point(location, size=0.1, color=carla.Color(0, 255, 0), life_time=2.0)
    
class BEVSample:
    '''
    Class for controlling the spectator camera and visualizing important map information in CARLA.

    This class is mainly used for debugging and scenario inspection. It can place the spectator
    camera above a target region, visualize vehicle and pedestrian spawn points, draw drivable
    waypoints, and display lane types across the map.

    Attributes:
        world: CARLA world object.
        world_map: CARLA map object used for waypoint queries and spawn point lookup.
        x, y, z: Spectator camera center position.
        dist: Distance used to define the local region of interest around the target location.
        wp_step: Step size used when generating waypoints for visualization.

    Methods:
        set_spector():
            Set the spectator camera to an overhead view centered at the target location.

        get_pos():
            Return the current spectator transform information.

        get_ped_spawn_points(ped_num, in_intersection=True):
            Generate pedestrian spawn points either inside the intersection region or across
            the whole map.

        show_waypoints():
            Draw drivable waypoints on the map for visualization.

        show_vehicles_spawn_points():
            Draw the default vehicle spawn points provided by the map.

        show_ped_spawn_points(ped_num, in_intersection=True):
            Draw sampled pedestrian spawn points for debugging.

        show_lane_types():
            Draw lane types across the map using different colors.

        show_intersection_info():
            Set the spectator view and draw multiple intersection-related debug visualizations.
    '''
    def __init__(self, actor: carla.Walker, bev_wrapper: BEVWrapper):
        self.wrapper = bev_wrapper
        self.actor = actor
        self.wrapper.hero_actor = actor
        self.feature_tensor = None

    def get_bev(self):
        # Focus the BEV engine on one pedestrian
        self.wrapper.hero_actor = self.actor
        
        # Extract the dictionary of layers
        layers = self.wrapper.get_bev_data()
        
        # 3. Stack them into a (320, 320, N) tensor 
        self.feature_tensor = np.stack([
            layers["lane"],
            layers["sidewalk"], 
            layers["shoulder"],
            layers["vehicle"], 
            layers["pedestrian"],
        ], axis=-1)
        
        return self.feature_tensor

    
    def visualize_bev(self):
        '''
        Get the output from self.get_bev_data() and visualize it as a single RGB image for debugging.
        lane: dark gray -> (40, 40, 40)
        sidewalk: gray -> (120, 120, 120)
        vehicle: red  -> (0, 0, 255)
        pedestrian: blue -> (255, 0, 0)
        hero pedestrian: green -> (0, 255, 0)

        HINT: OpenCV uses BRG format; Numpy/OpenCV images use (H, W)
        '''
        width = self.wrapper.width
        height = self.wrapper.height
        layers = self.wrapper.get_bev_data()
        self.wrapper.show_target_pedestrian()
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[layers["lane"] > 0] = (40, 40, 40)
        image[layers["sidewalk"] > 0] = (120, 120, 120)
        image[layers["vehicle"] > 0] = (0, 0, 255)
        image[layers["shoulder"] > 0] = (40, 0, 40)
        image[layers["pedestrian"] == 255] = (255, 0, 0)
        image[layers["pedestrian"] == 100] = (0, 255, 0)
        
        return image


# Functions for testing the functionality of BEV sampling
def find_pedestrian(world, test_ped):

    while test_ped is None:
        # world.tick() to ensure pedestrians are spawned
        world.tick() 
        ped_list = world.get_actors().filter("walker.*")   
        
        if len(ped_list) > 0:
            test_ped = ped_list[0]
            print(f"Find target pedestrian!! ID: {test_ped.id}")
        else:
            time.sleep(0.5)

    return test_ped

def BEV_test(world):
    bev_wrapper = BEVWrapper(cfg=None, world=world)
    ped_prev_location = {}
    ped_prev_frame = {}
    test_ped = None

    while True:
        world.tick()
        snapshot = world.get_snapshot()
        frame_id = snapshot.frame
        dt_fixed = world.get_settings().fixed_delta_seconds

        test_ped = find_pedestrian(world, test_ped)
        if not test_ped.is_alive:
            test_ped = None
            continue

        bev_sample = BEVSample(actor=test_ped, bev_wrapper=bev_wrapper)

        loc = test_ped.get_location()
        current_location = np.array([loc.x, loc.y], dtype=np.float32)

        if test_ped.id not in ped_prev_location:
            velocity = np.array([0.0, 0.0], dtype=np.float32)
        else:
            prev_location = ped_prev_location[test_ped.id]
            prev_frame = ped_prev_frame[test_ped.id]

            passed_frames = frame_id - prev_frame
            dt = passed_frames * dt_fixed

            if dt > 0:
                velocity = (current_location - prev_location) / dt
            else:
                velocity = np.array([0.0, 0.0], dtype=np.float32)

        
        speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
        ped_prev_location[test_ped.id] = current_location
        ped_prev_frame[test_ped.id] = frame_id

        print(f"Ped ID: {test_ped.id}")
        print(f"Location: {current_location}")
        print(f"Velocity: {velocity}")
        print(f"Speed: {speed}")

        image = bev_sample.visualize_bev()
        cv2.imshow("BEV Debug Tool", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    


if __name__ == "__main__":
    # Set up CARLA world and spawn actors as in intersection_sim.py
    client = carla.Client("localhost", 2000)
    world = client.get_world()
    
    # Test the functionality
    BEV_test(world)

