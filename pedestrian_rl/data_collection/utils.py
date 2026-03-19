import carla
import math
import random
import os
import h5py
import numpy as np
from ..utils.sim_utils import CrossroadPedestrians
from ..data_collection.bev.bev_sample import BEVSample, BEVWrapper
from ..data_collection.state_action_pair import PedestrianStateAction


class DataSampler:
    '''
    Class for sampling pedestrian state-action data during one simulation episode.

    This class manages the pedestrian data collection process for dataset generation.
    It keeps a fixed set of target pedestrian IDs for each episode, tracks their previous
    locations and frames, samples their current state and action, and stores the sampled
    results in an episode buffer.

    Attributes:
        world: CARLA world object.
        bev_wrapper: BEV wrapper used to generate bird's-eye-view observations.
        crossroad_pedestrians: CrossroadPedestrians object that stores spawned pedestrians and their goal locations.
        config: Configuration dictionary loaded from the config file.
        fixed_delta_time: Fixed simulation time step.
        sample_ped_num: Number of pedestrians to sample in each episode.
        prev_ped_location: Dictionary storing the previous location of each sampled pedestrian.
        prev_ped_frame: Dictionary storing the previous frame ID of each sampled pedestrian.
        episode_buffer: List storing sampled PedestrianStateAction objects for the current episode.
        target_ped_ids: List of sampled pedestrian IDs that remain fixed during the episode.

    Methods:
        append_sample(ped_info):
            Append one sampled PedestrianStateAction object to the episode buffer.

        get_episode_buffer():
            Return the sampled data buffer for the current episode.

        reset_episode_tracking():
            Reset previous pedestrian tracking info, episode buffer, and target pedestrian IDs
            when a new episode starts.

        select_target_ped_ids():
            Randomly select a fixed set of pedestrian IDs for the current episode.

        get_sample_pedestrians():
            Get the live pedestrian actor objects that correspond to the stored target IDs.

        sample_single_pedestrian(ped, frame_id, timestamp):
            Sample one pedestrian's state and action at the current frame, and return the
            PedestrianStateAction object together with its BEV sample.
    '''

    def __init__(self, world: carla.World, 
                 bev_wrapper: BEVWrapper, 
                 crossroad_pedestrians: CrossroadPedestrians, 
                 config
        ):

        self.world = world
        self.bev_wrapper = bev_wrapper
        self.crossroad_pedestrians = crossroad_pedestrians
        self.config = config

        self.fixed_delta_time = config["simulation"]["fixed_delta_seconds"]
        self.sample_ped_num = config["dataset"]["num_ped_per_episode"]

        self.prev_ped_location = {}
        self.prev_ped_frame = {}
        self.episode_buffer = []
        self.target_ped_ids = []

    def append_sample(self, ped_info):
        self.episode_buffer.append(ped_info)

    def get_episode_buffer(self):
        return self.episode_buffer
    
    def reset_episode_tracking(self):
        self.prev_ped_location = {}
        self.prev_ped_frame = {}
        self.episode_buffer = []
        self.target_ped_ids = []

    def select_target_ped_ids(self):
        all_peds = list(self.world.get_actors().filter("walker.*"))
        k = min(self.sample_ped_num, len(all_peds))

        if k == 0:
            self.target_ped_ids = []
        else:
            self.target_ped_ids = [ped.id for ped in random.sample(all_peds, k)]

        return self.target_ped_ids
    
    def get_sample_pedestrians(self):
        all_peds = list(self.world.get_actors().filter("walker.*"))
        ped_dict = {ped.id: ped for ped in all_peds}
        return [ped_dict[pid] for pid in self.target_ped_ids if pid in ped_dict]
    
    def sample_single_pedestrian(self, ped, frame_id, timestamp):
        bev_sample = BEVSample(actor=ped, bev_wrapper=self.bev_wrapper)
        controller = ped.get_control()
        ped_info = PedestrianStateAction(
            target_ped=ped,
            frame_id=frame_id,
            timestamp=timestamp
        )

        # ----- state -----
        bev_data = bev_sample.get_bev()

        current_carla_loc = ped.get_location()
        current_location = np.array(
            [current_carla_loc.x, current_carla_loc.y, current_carla_loc.z],
            dtype=np.float32
        )

        if ped.id not in self.prev_ped_location:
            velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            previous_location = self.prev_ped_location[ped.id]
            previous_frame = self.prev_ped_frame[ped.id]
            passed_frame = frame_id - previous_frame
            dt = passed_frame * self.fixed_delta_time

            if dt > 0:
                velocity = (current_location - previous_location) / dt
            else:
                velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        vx = velocity[0]
        vy = velocity[1]
        speed = float(math.sqrt(vx**2 + vy**2))

        if speed >= 0.1:
            motion_heading = math.atan2(vy, vx)
        else:
            motion_heading = math.radians(ped.get_transform().rotation.yaw)

        self.prev_ped_location[ped.id] = current_location
        self.prev_ped_frame[ped.id] = frame_id

        ped_goal_locations = self.crossroad_pedestrians.ped_goal_loc
        goal_location = ped_goal_locations.get(ped.id, current_location)

        ped_info.set_states(
            bev_data=bev_data,
            current_location=current_location,
            velocity=velocity,
            speed=speed,
            motion_heading=motion_heading,
            goal_location=goal_location
        )

        # ----- action -----
        target_speed = controller.speed
        direction = controller.direction
        target_direction = np.array(
            [direction.x, direction.y, direction.z],
            dtype=np.float32
        )

        ped_info.set_actions(
            target_speed=target_speed,
            target_direction=target_direction
        )

        return ped_info, bev_sample


    

def convert_to_dataset(episode_data: list, output_path, episode_idx=None):
    """
    Save one episode of pedestrian samples to an HDF5 file.

    Args:
        episode_data (list[PedestrianStateAction]):
            List of sampled pedestrian state-action objects from one episode.
        output_path (str):
            Path to output .h5 file.
        episode_idx (int | None):
            Optional episode index. If provided, data will be saved under
            group name f"episode_{episode_idx:05d}".
            If None, data is saved under "episode".

    HDF5 structure:
        /episode_xxxxx/
            ped_id              (N,)
            frame_id            (N,)
            timestamp           (N,)
            state/bev_data      (N, H, W, C)
            state/current_location  (N, 3)
            state/velocity      (N, 3)
            state/speed         (N,)
            state/motion_heading (N,)
            state/goal_location (N, 3)
            action/target_speed (N,)
            action/target_direction (N, 3)
    """
    if len(episode_data) == 0:
        print("[convert_to_dataset] No data to save.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    group_name = f"episode_{episode_idx:05d}" if episode_idx is not None else "episode"

    ped_ids = []
    frame_ids = []
    timestamps = []

    bev_data_list = []
    current_locations = []
    velocities = []
    speeds = []
    motion_headings = []
    goal_locations = []

    target_speeds = []
    target_directions = []

    for sample in episode_data:
        ped_ids.append(sample.ped_id)
        frame_ids.append(sample.state_action_pair["frame_id"])

        ts = sample.state_action_pair["timestamp"]
        # CARLA timestamp object -> float seconds
        if hasattr(ts, "elapsed_seconds"):
            timestamps.append(float(ts.elapsed_seconds))
        else:
            timestamps.append(float(ts))

        bev_data_list.append(np.asarray(sample.state["bev_data"], dtype=np.uint8))
        current_locations.append(np.asarray(sample.state["current_location"], dtype=np.float32))
        velocities.append(np.asarray(sample.state["velocity"], dtype=np.float32))
        speeds.append(np.float32(sample.state["speed"]))
        motion_headings.append(np.float32(sample.state["motion_heading"]))

        goal_loc = sample.state["goal_location"]
        if hasattr(goal_loc, "x"):  # carla.Location
            goal_loc = np.array([goal_loc.x, goal_loc.y, goal_loc.z], dtype=np.float32)
        else:
            goal_loc = np.asarray(goal_loc, dtype=np.float32)
        goal_locations.append(goal_loc)

        target_speeds.append(np.float32(sample.action["target_speed"]))
        target_directions.append(np.asarray(sample.action["target_direction"], dtype=np.float32))

    ped_ids = np.asarray(ped_ids, dtype=np.int32)
    frame_ids = np.asarray(frame_ids, dtype=np.int32)
    timestamps = np.asarray(timestamps, dtype=np.float64)

    bev_data = np.stack(bev_data_list, axis=0)                  # (N, H, W, C)
    current_locations = np.stack(current_locations, axis=0)     # (N, 3)
    velocities = np.stack(velocities, axis=0)                   # (N, 3)
    speeds = np.asarray(speeds, dtype=np.float32)               # (N,)
    motion_headings = np.asarray(motion_headings, dtype=np.float32)
    goal_locations = np.stack(goal_locations, axis=0)           # (N, 3)

    target_speeds = np.asarray(target_speeds, dtype=np.float32)
    target_directions = np.stack(target_directions, axis=0)     # (N, 3)

    with h5py.File(output_path, "a") as f:
        # overwrite same episode if exists
        if group_name in f:
            del f[group_name]

        grp = f.create_group(group_name)
        state_grp = grp.create_group("state")
        action_grp = grp.create_group("action")

        grp.create_dataset("ped_id", data=ped_ids)
        grp.create_dataset("frame_id", data=frame_ids)
        grp.create_dataset("timestamp", data=timestamps)

        state_grp.create_dataset("bev_data", data=bev_data, compression="gzip")
        state_grp.create_dataset("current_location", data=current_locations)
        state_grp.create_dataset("velocity", data=velocities)
        state_grp.create_dataset("speed", data=speeds)
        state_grp.create_dataset("motion_heading", data=motion_headings)
        state_grp.create_dataset("goal_location", data=goal_locations)

        action_grp.create_dataset("target_speed", data=target_speeds)
        action_grp.create_dataset("target_direction", data=target_directions)

    print(f"[convert_to_dataset] Saved {len(episode_data)} samples to {output_path}::{group_name}")


