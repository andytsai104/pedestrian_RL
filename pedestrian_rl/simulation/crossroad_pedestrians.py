import carla
import random
from ..utils.config_loader import load_config

class CrossroadPedestrians:
    '''
    Manages the spawning and AI control of pedestrians, allowing for them to be concentrated near a specific intersection or spread across the map.
    Pedestrians are spawned on the navigation mesh (with Z-offset) and given an AI controller to walk toward a random destination.

    Attributes:
        dist (float): Defines the bounding box distance from the center location for intersection-specific spawning.
        ped_num (int): The target number of pedestrians to spawn.
        speed (float): The maximum speed (m/s) for the walker AI controller. (maximum speed = 2)
    
    Methods:
        get_ped_spawn_points(): Generates a list of valid carla.Transform spawn points for pedestrians.
        spawn_single_walker(): Spawns a single walker and its AI controller, setting its destination.
        pedestrians_spawn(): Orchestrates the spawning of the total number of pedestrians.
    '''
    config = load_config("sim_config.json")["simulation"]
    def __init__(self, world, location,
                dist=config["intersection"]["dist"], 
                ped_num=config["pedestrian"]["ped_num"], 
                in_intersection=True
            ):
        
        self.world = world
        self.location = location
        self.x, self.y, self.z = location.x, location.y, location.z
        self.dist = dist
        self.ped_num = ped_num
        self.in_intersection = in_intersection
        self.speed = 1.0
        # self.speed = 2
    
    
    def get_ped_spawn_points(self, ped_num, in_intersection=True):
        spawn_points = []
        if in_intersection: # spawn within the intersection
            while len(spawn_points) < ped_num:
                '''
                Make sure to set spawn point (x, y, z=1) within intersection area
                spawn_point
                '''
                x_l = self.x - self.dist 
                x_h = self.x + self.dist
                y_l = self.y - self.dist
                y_h = self.y + self.dist
                spawn_point = carla.Transform()                
                spawn_point.location = self.world.get_random_location_from_navigation()
                spawn_point.location.z += 1.0
                if (x_l <= spawn_point.location.x <= x_h) and (y_l <= spawn_point.location.y <= y_h):
                    spawn_points.append(spawn_point)

        else: # spawn randomly on the map
            for _ in range(ped_num):
                spawn_point = carla.Transform()                
                spawn_point.location = self.world.get_random_location_from_navigation()
                spawn_point.location.z += 1.0
                spawn_points.append(spawn_point)

        return spawn_points


    def spawn_single_walker(self, spawn_location, destination):
        blueprint_library = self.world.get_blueprint_library()
        walker_bps = list(blueprint_library.filter('walker.pedestrian.*'))
        walker_bp = random.choice(walker_bps)
        
        if (type(spawn_location)==carla.Location):
            spawn_location = spawn_location + carla.Location(z=1)
            spawn_point = carla.Transform(spawn_location, carla.Rotation()) 
        elif (type(spawn_location) == carla.Transform):
            spawn_point = spawn_location
        else:
            spawn_point = None
            return False
        
        pedestrian = self.world.try_spawn_actor(walker_bp, spawn_point)

        self.world.tick()
        # time.sleep(0.05)

        # draw spawn point for visualization (Blue for spawn point)
        # self.world.debug.draw_point(spawn_point.location, size=0.2, color=carla.Color(0, 0, 255), life_time=-1)

        if pedestrian is None:
            # print("Failed to spawn pedestrian")
            return False

        controller_bp = blueprint_library.find('controller.ai.walker')
        controller = self.world.spawn_actor(controller_bp, carla.Transform(), pedestrian)


        if controller is None:
            # print("AI controller failed to spawn, destroying pedestrian.")
            pedestrian.destroy()
            return False

        controller.start()
        controller.set_max_speed(self.speed + random.random())
        controller.go_to_location(destination)

        return True

    
    def pedestrians_spawn(self):
        spawned = 0
        spawn_points = self.get_ped_spawn_points(self.ped_num, self.in_intersection)
        while spawned < self.ped_num:
            spawn_location = random.choice(spawn_points)
            destination = self.world.get_random_location_from_navigation()
            destination.z += 1
            if self.spawn_single_walker(spawn_location, destination):
                spawned += 1
