import time
import carla
from .utils import Spector, cleanup_simulation
from .aggressive_vehicles import AggressiveVehicles
from .crossroad_pedestrians import CrossroadPedestrians
import math

'''
TODO: Combine refreshing mechanism into classes (eg. merge ped. refreshing into CroossroadPedestrians, etc.)
'''

def refresh_simulation_if_needed(world, intersection_position: carla.Location, 
                                 aggressive_vehicles: AggressiveVehicles, cross_street_pedestrians: CrossroadPedestrians):
    # --- Refresh the simulation if vehicles are stuck or timeout occurs ---
    start_time = time.time()
    stuck_tracker = {}
    TIMEOUT_LIMIT = 60.0
    # Vehicles stcuk parameters
    STUCK_VELOCITY_THRESHOLD = 0.1  
    STUCK_TIME_LIMIT = 5.0          
    STUCK_VEHICLE_COUNT_LIMIT = 5
    # Minimum pedestrians in the intersection
    MIN_PEDESTRIANS_IN_INTERSECTION = 8
    INTERSECTION_DIST = 25   

    while True:
        world.tick()
        all_vehicles = world.get_actors().filter('vehicle.*')
        all_walkers = world.get_actors().filter('walker.*')
        current_time = time.time()
        elapsed_time = current_time - start_time
        stuck_count = 0
        ped_count = 0
        
        for vehicle in all_vehicles:
            v = vehicle.get_velocity()
            speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
            
            if speed < STUCK_VELOCITY_THRESHOLD:
                # Start timer if not already tracking
                if vehicle.id not in stuck_tracker:
                    stuck_tracker[vehicle.id] = current_time
                
                # Increment count only if stuck for consecutive 5+ seconds
                if (current_time - stuck_tracker[vehicle.id]) > STUCK_TIME_LIMIT:
                    stuck_count += 1
            else:
                # RESET: If the vehicle moves, remove it from tracker
                if vehicle.id in stuck_tracker:
                    del stuck_tracker[vehicle.id]
        
        for walker in all_walkers:
            walker_dist_to_intersection = walker.get_location().distance(intersection_position)
            # Check if walker is within the intersection area
            if walker_dist_to_intersection < INTERSECTION_DIST:
                ped_count += 1
        

        # Refresh the scenario if conditions are met
        reason = ""
        should_refresh = False
        if (elapsed_time > TIMEOUT_LIMIT):
            reason = "TIMEOUT"
            should_refresh = True

        elif (stuck_count >= STUCK_VEHICLE_COUNT_LIMIT):
            reason = f"DEADLOCK: {stuck_count} vehicles stuck"
            should_refresh = True

        elif ((elapsed_time > 5.0) and (ped_count <= MIN_PEDESTRIANS_IN_INTERSECTION)):
            reason = f"LOW PEDESTRIANS: only {ped_count} pedestrians in intersection"
            should_refresh = True

        if should_refresh:
            print(f"--- Round Ended: {reason}. ---")
            print("Refreshing simulation...")
            
            # Cleanup pedestrians and vehivles
            cleanup_simulation(world)

            # Re-spawn vehicles and pedestrians
            aggressive_vehicles.aggressive_vehicles_spawn()
            cross_street_pedestrians.pedestrians_spawn()

            # Reset timer and tracker
            start_time = time.time()
            stuck_tracker = {}

        time.sleep(0.05)


def main():
    # --- Initialize ---
    # CARLA Server
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.load_world('Town10HD_Opt')
    # world = client.get_world()
    world_map = world.get_map()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    intersection_position = carla.Location(x=-43.5, y=21, z=0)

    # Set spector to visulize intersection area
    spector = Spector(world, location=intersection_position + carla.Location(z=50), dist=25)
    # spector.show_intersection_info()
    spector.set_spector()

    # Spawn aggressive vehicles
    aggressive_vehicles = AggressiveVehicles(client, world, world_map, location=intersection_position)
    aggressive_vehicles.aggressive_vehicles_spawn()

    # Spawn crossroad pedestrians
    cross_street_pedestrians = CrossroadPedestrians(world, location=intersection_position)
    cross_street_pedestrians.pedestrians_spawn()

    refresh_simulation_if_needed(world, intersection_position, aggressive_vehicles, cross_street_pedestrians)

if __name__ == '__main__':
    main()

    