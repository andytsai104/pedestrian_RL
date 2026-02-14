import carla
import time
import numpy as np

'''Safely destroys pedestrians and vehicles.'''
def cleanup_simulation(world):
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
        dist = 25
        all_points = []
        
        waypoints = self.world_map.generate_waypoints(distance=step_size)

        # Extract all X and Y coordinates
        x_coords = [w.transform.location.x for w in waypoints]
        y_coords = [w.transform.location.y for w in waypoints]

        x_max = int(max(x_coords) + dist)
        x_min = int(min(x_coords) - dist)
        y_max = int(max(y_coords) + dist)
        y_min = int(min(y_coords) - dist)

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


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    location = carla.Location(x=-43.5, y=21, z=50)
    spector = Spector(world, location)
    spector.set_spector()
    # spector.show_intersection_info()
    spector.show_lane_types()
    
    while True:
        world.tick()
        time.sleep(0.1)

if __name__ == "__main__":
    main()


