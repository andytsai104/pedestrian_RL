import carla
import time
import numpy as np
import math
from .config_loader import load_config
import random


class Spector:
    '''
    Manages the spectator camera and provides debugging visualization tools for displaying key map features in CARLA.
    Initializes the camera position and defines boundaries (dist) for drawing waypoints and vehicle/pedestrian spawn points.

    Methods:
        set_spector(): Positions the camera for an overhead view.
        get_pos(): Get the current spector's coordinate.
        get_ped_spawn_points(): Get a list of pedestrians' spawn points.
        show_waypoints(): Draws map waypoints (green). -> only drivable points
        show_vehicles_spawn_points(): Draws default vehicle spawn points (red).
        show_ped_spawn_points(): Draws pedestrian spawn points inside/outside the intersection (blue).
        show_lane_types(): Draw all lane types on the map and set spector to the center of the map.
        show_intersection_info(): Sets camera and draws all visualizations on the intersection.
    '''
    def __init__(self, world, location, dist=25, wp_step=2):
        self.dist = dist
        self.wp_step = wp_step
        self.x, self.y, self.z = location.x, location.y, location.z
        self.world = world
        self.world_map = self.world.get_map()


    def set_spector(self):
        spectator = self.world.get_spectator()
        spectator.set_transform(carla.Transform(
            carla.Location(self.x, self.y, self.z),
            carla.Rotation(yaw=-90, pitch=-90, roll=0)
        ))

    def get_pos(self) -> dict:
        t = self.world.get_spectator().get_transform()
        loc, rot = t.location, t.rotation
        # print(f"x={loc.x:.2f}, y={loc.y:.2f}, z={loc.z:.2f}, yaw={rot.yaw:.1f}")

        return {"Location": loc, "Rotation": rot}
    
    def get_ped_spawn_points(self, ped_num, in_intersection=True) -> list:
        spawn_points = []
        if in_intersection:     # spawn within the intersection
            while len(spawn_points) < ped_num:
                # make sure that spawn point (x, y, z=1) within intersection area
                x_l = self.x - self.dist 
                x_h = self.x + self.dist
                y_l = self.y - self.dist
                y_h = self.y + self.dist
                spawn_point = carla.Transform()                
                spawn_point.location = self.world.get_random_location_from_navigation()
                spawn_point.location.z += 1.0
                if (x_l <= spawn_point.location.x <= x_h) and (y_l <= spawn_point.location.y <= y_h):
                    spawn_points.append(spawn_point)

        else:       # spawn randomly on the map
            for _ in range(ped_num):
                spawn_point = carla.Transform()                
                spawn_point.location = self.world.get_random_location_from_navigation()
                spawn_points.append(spawn_point)

        return spawn_points

    def show_waypoints(self):
        # Possible vehicles spawn points
        waypoints = self.world_map.generate_waypoints(self.wp_step)
        for w in waypoints:
            wp = w.transform.location + carla.Location(z=1.0)
            self.world.debug.draw_point(wp, size=0.1, color=carla.Color(0, 255, 0), life_time=-1)

    def show_vehicles_spawn_points(self):
        spawn_points = self.world_map.get_spawn_points()
        for sp in spawn_points:
            spawn_point = sp.location + carla.Location(z=1.0)
            self.world.debug.draw_point(spawn_point, size=0.1, color=carla.Color(255, 0, 0), life_time=-1)

    def show_ped_spawn_points(self, ped_num, in_intersection=True):
        spawn_points = self.get_ped_spawn_points(ped_num, in_intersection)
        # draw spawn points    
        for spawn_point in spawn_points:
            self.world.debug.draw_point(spawn_point.location, size=0.1, color=carla.Color(0, 0, 255), life_time=-1)


    def show_lane_types(self):
        # Draw all lane types on the map based on step size from (min_waypoint_coordinates - dist) to (max_waypoint_coordinates + dist).
        
        original_coor = (self.x, self.y, self.z)
        # Define colors for each lane types
        LANE_COLORS = {
            carla.LaneType.NONE: carla.Color(50, 50, 50),            # Dark Gray
            carla.LaneType.Driving: carla.Color(0, 150, 255),        # Light Blue
            carla.LaneType.Stop: carla.Color(255, 0, 0),             # Red
            carla.LaneType.Shoulder: carla.Color(255, 255, 0),       # Yellow
            carla.LaneType.Biking: carla.Color(0, 255, 255),         # Cyan
            carla.LaneType.Sidewalk: carla.Color(0, 255, 0),         # Green
            carla.LaneType.Border: carla.Color(139, 69, 19),         # Saddle Brown
            carla.LaneType.Restricted: carla.Color(128, 0, 0),       # Maroon
            carla.LaneType.Parking: carla.Color(255, 165, 0),        # Orange
            carla.LaneType.Bidirectional: carla.Color(200, 150, 255),# Light Purple
            carla.LaneType.Median: carla.Color(46, 139, 87),         # Sea Green
            carla.LaneType.Special1: carla.Color(238, 130, 238),     # Violet
            carla.LaneType.Special2: carla.Color(255, 20, 147),      # Deep Pink
            carla.LaneType.Special3: carla.Color(218, 112, 214),     # Orchid
            carla.LaneType.RoadWorks: carla.Color(255, 69, 0),       # Red-Orange
            carla.LaneType.Tram: carla.Color(72, 61, 139),           # Dark Slate Blue
            carla.LaneType.Rail: carla.Color(169, 169, 169),         # Medium Gray
            carla.LaneType.Entry: carla.Color(152, 251, 152),        # Pale Green
            carla.LaneType.Exit: carla.Color(255, 218, 185),         # Peach
            carla.LaneType.OffRamp: carla.Color(250, 128, 114),      # Salmon
            carla.LaneType.OnRamp: carla.Color(127, 255, 212)        # Aquamarine
        }
        step_size = 0.5
        mark_size = 0.1
        all_points = []
        
        waypoints = self.world_map.generate_waypoints(distance=step_size)

        # Extract all X and Y coordinates
        x_coords = [w.transform.location.x for w in waypoints]
        y_coords = [w.transform.location.y for w in waypoints]

        x_max = int(max(x_coords) + self.dist)
        x_min = int(min(x_coords) - self.dist)
        y_max = int(max(y_coords) + self.dist)
        y_min = int(min(y_coords) - self.dist)

        # Set spector to the center of the map
        self.x, self.y, self.z = (x_max - x_min) // 2, (y_max - y_min) // 2, 150
        self.set_spector()
        self.x, self.y, self.z = original_coor[0], original_coor[1], original_coor[2]

        # Get all points on the map based on step size
        for x_pos in np.arange(x_min, x_max, step_size):
            for y_pos in np.arange(y_min, y_max, step_size):
                all_points.append(carla.Location(x=float(x_pos), y=float(y_pos), z=5.0))
        
        # Draw the lane_type on the map
        for loc_point in all_points:
            wp = self.world_map.get_waypoint(loc_point, project_to_road=False, lane_type=carla.LaneType.Any)
            if wp:
                lane_type = wp.lane_type
                color = LANE_COLORS.get(lane_type, carla.Color(255, 255, 255))

                location = carla.Location(x=loc_point.x, y=loc_point.y, z=5.0)
                location.z = 0.5

                self.world.debug.draw_point(location, size = mark_size, color = color, life_time = -1)


    def show_intersection_info(self):
        self.set_spector()
        # Draw map info
        self.show_waypoints()
        self.show_vehicles_spawn_points()
        self.show_ped_spawn_points(ped_num=1000)


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


class AggressiveVehicles:
    '''
    Manages the spawning and configuration of vehicles with aggressive driving behaviors near a specific intersection.
    
    Vehicles are spawned on driving lanes within a specified distance of the target location 
    and are configured to ignore traffic lights/signs and exceed the speed limit by 100% via the CARLA Traffic Manager.

    Attributes:
        veh_num (int): The target number of vehicles to spawn.
        dist_to_intersection (float): Maximum distance (meters) from the location to find spawn points.
        speed_diff (float): Percentage speed difference (negative is faster than limit).
        dist_lead (float): Minimum distance to maintain from a leading vehicle.
        wp_step (int): 
    
    Methods:
        aggressive_vehicles_spawn(): Finds spawn points, spawns vehicles, and applies Traffic Manager settings for aggressive behavior.

    
    TODO Check if TM has ignore ped or not (improve vehicle performance)

    '''
    config = load_config("sim_config.json")["simulation"]["vehicle"]
    def __init__(self, client, world, world_map, location, 
                 veh_num=config["veh_num"], 
                 dist_to_intersection=config["dist_to_intersection"],
                 speed_diff=config["speed_diff"],
                 dist_lead=config["dist_lead"], 
                 wp_step=config["wp_step"]
        ):
        
        self.client = client
        self.world = world
        self.world_map = world_map
        self.location = location
        self.veh_num = veh_num
        self.dist_to_intersection = dist_to_intersection
        self.speed_diff = speed_diff
        self.dist_lead = dist_lead
        self.wp_step = wp_step

    def aggressive_vehicles_spawn(self):
        # Traffic manager
        tm = self.client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)

        # Spawning cars around the intersection
        blueprint_library = self.world.get_blueprint_library()
        # Possible vehicles spawn points
        waypoints = self.world_map.generate_waypoints(self.wp_step)
        vehicles_bp = [bp for bp in blueprint_library.filter('vehicle.*') 
                    if bp.get_attribute('base_type') == 'car']              # get vehicles with base_type='car'
        vehicle_bp = [random.choice(vehicles_bp) for _ in range(self.veh_num)]
        veh_wps = []
        for wp in waypoints:
            # Only spawn on driving lanes
            if wp.lane_type != carla.LaneType.Driving:
                continue
            dist = wp.transform.location.distance(self.location)
            wp_tf = wp.transform
            # Select waypoints within a certain distance to the intersection
            if dist < self.dist_to_intersection:
                spawn_pos = carla.Transform(wp_tf.location + carla.Location(z=1.0), wp_tf.rotation)
                veh_wps.append(spawn_pos)


        # Spawn vehicles and apply TM settings
        random.shuffle(veh_wps)
        veh_spawn_points = veh_wps[:self.veh_num]
        vehicles = []
        for i in range(self.veh_num):
            veh = self.world.try_spawn_actor(random.choice(vehicle_bp), veh_spawn_points[i])
            if veh:
                vehicles.append(veh)
                veh.set_autopilot(True, tm.get_port())


        for v in vehicles:
            tm.vehicle_percentage_speed_difference(v, self.speed_diff)
            tm.distance_to_leading_vehicle(v, self.dist_lead)
            tm.ignore_lights_percentage(v, 100)             # ignore traffic lights 100%
            tm.ignore_signs_percentage(v, 100)              # ignore traffic signs 100%


def cleanup_simulation(world):
    '''Safely destroys pedestrians and vehicles.'''

    # Get all relevant actors
    vehicles = world.get_actors().filter('vehicle.*')
    walkers = world.get_actors().filter('walker.*')
    controllers = world.get_actors().filter('controller.ai.*')
    
    # Stop controllers first to prevent them from trying to control dead walkers
    for controller in controllers:
        if controller.is_alive:
            controller.stop()
            controller.destroy()

    # Destroy walkers and vehicles
    for actor in list(walkers) + list(vehicles):
        if actor.is_alive:
            actor.destroy()
    
    # Tick the world to let the server process the destructions
    world.tick()


def spawn_actors(world, aggressive_vehicles: AggressiveVehicles, cross_street_pedestrians: CrossroadPedestrians):
    
    # Cleanup pedestrians and vehivles
    cleanup_simulation(world)

    # Spawn vehicles and pedestrians
    aggressive_vehicles.aggressive_vehicles_spawn()
    cross_street_pedestrians.pedestrians_spawn()


def refresh_sim(world, refresh_conditions: dict, intersection_position: carla.Location):
    
    all_vehicles = world.get_actors().filter("vehicle.*")
    all_walkers = world.get_actors().filter("walker.*")
    current_time = world.get_snapshot().timestamp.elapsed_seconds
    veh_stuck_count = 0
    ped_count = 0

    # Refresh conditions
    start_time = refresh_conditions["start time"]
    time_out = refresh_conditions["time_out"]
    veh_stuck_tracker = refresh_conditions["vehicle"]["stuck_tracker"]
    veh_stuck_time_limit = refresh_conditions["vehicle"]["stuck_time_limit"]
    veh_stuck_count_limit = refresh_conditions["vehicle"]["stuck_count_limit"]
    veh_vel_thres = refresh_conditions["vehicle"]["velocity_threshold"]
    min_peds = refresh_conditions["pedestrian"]["min_peds"]
    min_ped_dist = refresh_conditions["pedestrian"]["dist"]

    elapsed_time = current_time - start_time

    # Get stuck vehicles
    for vehicle in all_vehicles:
        v = vehicle.get_velocity()
        speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
        
        if speed < veh_vel_thres:
            # Start timer if not already tracking
            if vehicle.id not in veh_stuck_tracker:
                veh_stuck_tracker[vehicle.id] = current_time
            
            # Increment count only if stuck over the condition (s)
            if (current_time - veh_stuck_tracker[vehicle.id]) > veh_stuck_time_limit:
                veh_stuck_count += 1
        else:
            # RESET: If the vehicle moves, remove it from tracker
            if vehicle.id in veh_stuck_tracker:
                del veh_stuck_tracker[vehicle.id]
    
    # Get pedestrian count in the intersection
    for walker in all_walkers:
        walker_dist_to_intersection = walker.get_location().distance(intersection_position)
        # Check if walker is within the intersection area
        if walker_dist_to_intersection < min_ped_dist:
            ped_count += 1


    # Determine whether refreshing
    sim_state = ""
    should_refresh = False
    
    if elapsed_time > time_out:
        sim_state = "Time Out!"
        should_refresh = True
    
    elif veh_stuck_count > veh_stuck_count_limit:
        sim_state = f"DEADLOCK: {veh_stuck_count_limit} vehicles stuck!"
        should_refresh = True

    elif (elapsed_time > 3) and (ped_count <= min_peds):
        sim_state = f"LOW PEDESTRIANS: Only {ped_count} pedestrians in the intersection!"
        should_refresh = True

    return sim_state, should_refresh