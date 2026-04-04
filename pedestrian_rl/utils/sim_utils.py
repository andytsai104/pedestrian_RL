import carla
import time
import numpy as np
import math
from .config_loader import load_config
import random

class Spector:
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
    Class for spawning and managing pedestrians in the intersection scenario.

    This class generates pedestrian spawn points, spawns walkers with AI controllers, assigns
    walking destinations, and keeps track of each pedestrian's goal location. Pedestrians can
    either be concentrated near the target intersection or sampled from the whole map.

    Attributes:
        world: CARLA world object.
        location: Center location of the target intersection.
        x, y, z: Coordinates of the target intersection center.
        dist: Distance used to define the intersection spawning region.
        ped_num: Number of pedestrians to spawn.
        in_intersection: Whether pedestrians should be spawned only near the intersection.
        speed: Walking speed assigned to the pedestrian AI controller.
        ped_goal_loc: Dictionary mapping pedestrian ID to its goal location.

    Methods:
        get_ped_spawn_points(ped_num, in_intersection=True):
            Generate valid pedestrian spawn points either inside the intersection area or
            randomly across the map.

        spawn_single_walker(spawn_location, destination):
            Spawn one pedestrian and its AI controller, assign a destination, and store the
            pedestrian's goal location.

        pedestrians_spawn():
            Spawn the full set of pedestrians for the current episode.

        reset_pedestrians():
            Clear the stored pedestrian goal-location dictionary.
    '''
    config = load_config("sim_config.json")["simulation"]
    def __init__(self, world, location,
                dist=config["intersection"]["dist"], 
                ped_num=config["pedestrian"]["ped_num"],
                speed_range=config["pedestrian"]["speed_range"], 
                in_intersection=True
            ):
        
        self.world = world
        self.location = location
        self.x, self.y, self.z = location.x, location.y, location.z
        self.dist = dist
        self.ped_num = ped_num
        self.in_intersection = in_intersection
        self.speed_range = speed_range
        
        self.ped_controller = {}
        self.ped_controller_type = {}
        self.ped_goal_loc = {}
    
    
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


    def spawn_single_walker(
            self,
            spawn_location,
            destination: carla.Location,
            controller="ai",
            max_speed: float = None,
        ):
        blueprint_library = self.world.get_blueprint_library()
        walker_bps = list(blueprint_library.filter("walker.pedestrian.*"))
        walker_bp = random.choice(walker_bps)

        if isinstance(spawn_location, carla.Location):
            spawn_location = spawn_location + carla.Location(z=1.0)
            spawn_point = carla.Transform(spawn_location, carla.Rotation())
        elif isinstance(spawn_location, carla.Transform):
            spawn_point = spawn_location
        else:
            return None

        pedestrian = self.world.try_spawn_actor(walker_bp, spawn_point)
        self.world.tick()

        if pedestrian is None:
            return None

        ped_speed = (
            float(max_speed)
            if max_speed is not None
            else random.uniform(self.speed_range[0], self.speed_range[1])
        )

        controller_ref = None
        controller_type = None

        # ---- built-in CARLA AI walker controller ----
        if isinstance(controller, str) and controller == "ai":
            controller_bp = blueprint_library.find("controller.ai.walker")
            controller_actor = self.world.spawn_actor(
                controller_bp,
                carla.Transform(),
                pedestrian
            )

            if controller_actor is None:
                pedestrian.destroy()
                self.world.tick()
                return None

            controller_actor.start()
            controller_actor.set_max_speed(ped_speed)
            controller_actor.go_to_location(destination)

            controller_ref = controller_actor
            controller_type = "ai"

        # ---- no AI controller attached; external code will apply_control() ----
        elif controller is None or (isinstance(controller, str) and controller == "manual"):
            controller_ref = None
            controller_type = "manual"

        # ---- external Python-side controller object, e.g. BC runner / RL runner ----
        else:
            controller_ref = controller
            controller_type = "external"

        self.ped_goal_loc[pedestrian.id] = np.array(
            [destination.x, destination.y, destination.z],
            dtype=np.float32
        )
        self.ped_controller[pedestrian.id] = controller_ref
        self.ped_controller_type[pedestrian.id] = controller_type

        return pedestrian


    def reset_pedestrians(self):
        self.ped_goal_loc = {}
        self.ped_controller = {}
        self.ped_controller_type = {}

    
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
    Class for spawning vehicles with aggressive driving behavior near the target intersection.

    This class finds nearby drivable spawn locations, spawns vehicles, enables autopilot,
    and configures the CARLA Traffic Manager so that the spawned vehicles behave more
    aggressively, such as driving faster and ignoring traffic lights or signs.

    Attributes:
        client: CARLA client object.
        world: CARLA world object.
        world_map: CARLA map object used for waypoint queries.
        location: Center location of the target intersection.
        veh_num: Number of vehicles to spawn.
        dist_to_intersection: Maximum distance from the intersection for valid vehicle spawn points.
        speed_diff: Percentage speed difference applied through the Traffic Manager.
        dist_lead: Desired distance to the leading vehicle.
        wp_step: Step size used when generating waypoints for candidate vehicle spawn points.

    Methods:
        aggressive_vehicles_spawn():
            Generate nearby vehicle spawn points, spawn vehicles, enable autopilot, and apply
            aggressive Traffic Manager settings.
    '''
    config = load_config("sim_config.json")["simulation"]["vehicle"]
    def __init__(self, client, world, location, 
                 veh_num=config["veh_num"], 
                 dist_to_intersection=config["dist_to_intersection"],
                 speed_diff=config["speed_diff"],
                 dist_lead=config["dist_lead"], 
                 wp_step=config["wp_step"]
        ):
        
        self.client = client
        self.world = world
        self.world_map = self.world.get_map()
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


def spawn_actors(world, 
                 spector: Spector, 
                 aggressive_vehicles: AggressiveVehicles,
                 crossroad_pedestrians: CrossroadPedestrians,
    ):
    # Clean up world
    cleanup_simulation(world)
    crossroad_pedestrians.reset_pedestrians()

    # Set spector
    spector.set_spector()
    # Spawn actors
    aggressive_vehicles.aggressive_vehicles_spawn()
    crossroad_pedestrians.pedestrians_spawn()

    # Warmup world
    for _ in range(3):
        world.tick()