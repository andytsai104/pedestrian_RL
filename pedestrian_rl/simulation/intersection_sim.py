import time
import carla
from ..utils.intersection_sim_utils import Spector, cleanup_simulation
from ..utils.config_loader import load_config
from .aggressive_vehicles import AggressiveVehicles
from .crossroad_pedestrians import CrossroadPedestrians
import math

def refresh_simulation_if_needed(world, config, dist, intersection_position: carla.Location, 
                                 aggressive_vehicles: AggressiveVehicles, cross_street_pedestrians: CrossroadPedestrians):
    # --- Refresh the simulation if vehicles are stuck or timeout occurs ---
    start_time = time.time()
    stuck_tracker = {}
    time_out = config["time_out"]
    min_peds = config["pedestrian"]["min_pedestrians"]
    veh_vel_threshold = config["vehicle"]["velocity_threshold"]
    veh_stuck_time = config["vehicle"]["stuck_time_limit"]
    veh_stuck_count = config["vehicle"]["stuck_count_limit"]

    while True:
        world.tick()
        all_vehicles = world.get_actors().filter("vehicle.*")
        all_walkers = world.get_actors().filter("walker.*")
        current_time = time.time()
        elapsed_time = current_time - start_time
        stuck_count = 0
        ped_count = 0
        
        for vehicle in all_vehicles:
            v = vehicle.get_velocity()
            speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
            
            if speed < veh_vel_threshold:
                # Start timer if not already tracking
                if vehicle.id not in stuck_tracker:
                    stuck_tracker[vehicle.id] = current_time
                
                # Increment count only if stuck for consecutive 5+ seconds
                if (current_time - stuck_tracker[vehicle.id]) > veh_stuck_time:
                    stuck_count += 1
            else:
                # RESET: If the vehicle moves, remove it from tracker
                if vehicle.id in stuck_tracker:
                    del stuck_tracker[vehicle.id]
        
        for walker in all_walkers:
            walker_dist_to_intersection = walker.get_location().distance(intersection_position)
            # Check if walker is within the intersection area
            if walker_dist_to_intersection < dist:
                ped_count += 1
        

        # Refresh the scenario if conditions are met
        reason = ""
        should_refresh = False
        if (elapsed_time > time_out):
            reason = "TIMEOUT"
            should_refresh = True

        elif (stuck_count >= veh_stuck_count):
            reason = f"DEADLOCK: {stuck_count} vehicles stuck"
            should_refresh = True

        elif ((elapsed_time > 5.0) and (ped_count <= min_peds)):
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

        # time.sleep(0.05)


def main():
    # Load config
    config = load_config("sim_config.json")
    sim_config = config["simulation"]
    # --- Initialize ---
    # CARLA Server
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.load_world(sim_config["town"])
    # world = client.get_world()
    world_map = world.get_map()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = sim_config["fixed_delta_seconds"]
    world.apply_settings(settings)

    intersection_position = carla.Location(
        x=sim_config["intersection"]["x"],
        y=sim_config["intersection"]["y"], 
        z=sim_config["intersection"]["z"],
    )

    distance = sim_config["intersection"]["dist"]

    # Set spector to visulize intersection area
    spector = Spector(world, location=intersection_position + carla.Location(z=50), dist=distance)
    # spector.show_intersection_info()
    spector.set_spector()

    # Spawn aggressive vehicles
    aggressive_vehicles = AggressiveVehicles(client, world, world_map, location=intersection_position)
    aggressive_vehicles.aggressive_vehicles_spawn()

    # Spawn crossroad pedestrians
    cross_street_pedestrians = CrossroadPedestrians(world, location=intersection_position)
    cross_street_pedestrians.pedestrians_spawn()

    stuck_detection_config = sim_config["stuck_detection"]
    refresh_simulation_if_needed(world=world, config=stuck_detection_config, dist=distance, intersection_position=intersection_position, aggressive_vehicles=aggressive_vehicles, cross_street_pedestrians=cross_street_pedestrians)

if __name__ == "__main__":
    main()

    