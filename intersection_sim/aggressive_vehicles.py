import carla
import random


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
        aggressive_vehicles_spawn(): Finds spawn points, spawns vehicles, and applies 
                                     Traffic Manager settings for aggressive behavior.
    '''
    def __init__(self, client, world, world_map, location, veh_num=30, 
                 dist_to_intersection=60, speed_diff=-100, dist_lead=2.0, wp_step=2):
        
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


