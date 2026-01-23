import carla
import time

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
        show_waypoints(): Draws map waypoints (green).
        show_vehicles_spawn_points(): Draws default vehicle spawn points (red).
        show_ped_spawn_points(): Draws pedestrian spawn points inside/outside the intersection (blue).
        show_intersection_info(): Sets camera and draws all visualizations.
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

    def get_pos(self):
        t = self.world.get_spectator().get_transform()
        loc, rot = t.location, t.rotation
        print(f"x={loc.x:.2f}, y={loc.y:.2f}, z={loc.z:.2f}, yaw={rot.yaw:.1f}")

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

    def get_ped_spawn_points(self, ped_num, in_intersection=True):
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

    def show_ped_spawn_points(self, ped_num, in_intersection=False):
        spawn_points = self.get_ped_spawn_points(ped_num, in_intersection)
        # draw spawn points    
        for spawn_point in spawn_points:
            self.world.debug.draw_point(spawn_point.location, size=0.1, color=carla.Color(0, 0, 255), life_time=-1)


    def show_intersection_info(self):
        self.set_spector()
        # Draw map info
        self.show_waypoints()
        self.show_vehicles_spawn_points()
        self.show_ped_spawn_points(ped_num=100, in_intersection=True)



if __name__ == "__main__":
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    static = True
    location = carla.Location(x=-43.5, y=21, z=50)

    spector = Spector(world, location)
    spector.show_intersection_info()

    while True:
        world.tick()
        time.sleep(0.1)

