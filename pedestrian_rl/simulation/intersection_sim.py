import time
import carla
from ..utils.sim_utils import Spector, CrossroadPedestrians, AggressiveVehicles, cleanup_simulation, refresh_sim
from ..utils.config_loader import load_config    


def refresh_simulation_if_needed(world: carla.World, config, intersection_position: carla.Location, 
                                 aggressive_vehicles: AggressiveVehicles, crossroad_pedestrians: CrossroadPedestrians):
    # --- Refresh the simulation if vehicles are stuck or timeout occurs ---
    refresh_conditions = {
        "time_out": config["time_out"],
        "start time": world.get_snapshot().timestamp.elapsed_seconds,
        "vehicle": {
            "velocity_threshold": config["vehicle"]["velocity_threshold"],
            "stuck_tracker": {},
            "stuck_time_limit": config["vehicle"]["stuck_time_limit"],
            "stuck_count_limit": config["vehicle"]["stuck_count_limit"],
        },
        "pedestrian": {
            "dist": config["pedestrian"]["dist"],
            "min_peds": config["pedestrian"]["min_pedestrians"],
        },
    }
    while True:
        world.tick()        
        sim_state, should_refresh = refresh_sim(world=world, refresh_conditions=refresh_conditions, intersection_position=intersection_position)

        if should_refresh:
            print(f"--- {sim_state} ---")
            print("Refreshing simulation...")
            
            # Cleanup pedestrians and vehivles
            cleanup_simulation(world)

            # Reset timer and tracker
            refresh_conditions["start time"] = world.get_snapshot().timestamp.elapsed_seconds
            refresh_conditions["vehicle"]["stuck_tracker"] = {}

            crossroad_pedestrians.reset_pedestrians()

            # Spawn vehicles and pedestrians
            aggressive_vehicles.aggressive_vehicles_spawn()
            crossroad_pedestrians.pedestrians_spawn()


def main():
    # Load config
    config = load_config("sim_config.json")
    sim_config = config["simulation"]
    # --- Initialize ---
    # CARLA Server
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.load_world(sim_config["town"])
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = sim_config["fixed_delta_seconds"]
    world.apply_settings(settings)

    intersection_position = carla.Location(
        x=sim_config["intersection"]["x"],
        y=sim_config["intersection"]["y"], 
        z=sim_config["intersection"]["z"],
    )
    spector_height = sim_config["intersection"]["spector_height"]
    distance = sim_config["intersection"]["dist"]

    # Set spector to visulize intersection area
    spector = Spector(world, location=intersection_position + carla.Location(z=spector_height), dist=distance)
    # spector.show_intersection_info()
    spector.set_spector()

    # Spawn aggressive vehicles
    aggressive_vehicles = AggressiveVehicles(client, world, location=intersection_position)

    # Spawn crossroad pedestrians
    crossroad_pedestrians = CrossroadPedestrians(world, location=intersection_position)
    
    aggressive_vehicles.aggressive_vehicles_spawn()
    crossroad_pedestrians.pedestrians_spawn()

    stuck_detection_config = sim_config["stuck_detection"]
    refresh_simulation_if_needed(world=world, config=stuck_detection_config, intersection_position=intersection_position, aggressive_vehicles=aggressive_vehicles, crossroad_pedestrians=crossroad_pedestrians)
    
if __name__ == "__main__":
    main()

    