import carla
import math
import random
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from ..utils.sim_utils import CrossroadPedestrians
from .bev.bev_sample import BEVSample, BEVWrapper
from .state_action_pair import PedestrianStateAction


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
            /ped_<id>/
                frame_id                    (N,)
                timestamp                   (N,)
                state/bev_data              (N, H, W, C)
                state/current_location      (N, 3)
                state/velocity              (N, 3)
                state/speed                 (N,)
                state/motion_heading        (N,)
                state/goal_location         (N, 3)
                action/target_speed         (N,)
                action/target_direction     (N, 3)
    """
    if len(episode_data) == 0:
        print("[convert_to_dataset] No data to save.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    group_name = f"episode_{episode_idx:03d}" if episode_idx is not None else "episode"

    # Organize the sampled data to {ped_id: [PedestrainStateAction_1, PedestrainStateAction_2, ...]}
    ped_groups = defaultdict(list)
    for sample in episode_data:
        ped_groups[sample.ped_id].append(sample)

    with h5py.File(output_path, "a") as file:
        # overwrite same episode if exists
        if group_name in file:
            del file[group_name]

        episode_grp = file.create_group(group_name)

        for ped_id, samples in ped_groups.items():
            # sort each pedestrian's samples by frame_id
            samples = sorted(samples, key=lambda s: s.state_action_pair["frame_id"])

            # --- Info Buffer Initialization ---
            # Time
            frame_ids = []
            timestamps = []
            
            # States
            bev_data_list = []
            current_locations = []
            velocities = []
            speeds = []
            motion_headings = []
            goal_locations = []

            # Actions
            target_speeds = []
            target_directions = []

            # --- Store Data ---
            for sample in samples:
                # Time
                frame_ids.append(sample.state_action_pair["frame_id"])

                ts = sample.state_action_pair["timestamp"]
                if hasattr(ts, "elapsed_seconds"):
                    timestamps.append(float(ts.elapsed_seconds))
                else:
                    timestamps.append(float(ts))

                # States
                bev_data_list.append(np.asarray(sample.state["bev_data"], dtype=np.uint8))
                current_locations.append(np.asarray(sample.state["current_location"], dtype=np.float32))
                velocities.append(np.asarray(sample.state["velocity"], dtype=np.float32))
                speeds.append(np.float32(sample.state["speed"]))
                motion_headings.append(np.float32(sample.state["motion_heading"]))

                goal_loc = sample.state["goal_location"]
                if hasattr(goal_loc, "x"):   # carla.Location
                    goal_loc = np.array([goal_loc.x, goal_loc.y, goal_loc.z], dtype=np.float32)
                else:
                    goal_loc = np.asarray(goal_loc, dtype=np.float32)
                goal_locations.append(goal_loc)

                # Actions
                target_speeds.append(np.float32(sample.action["target_speed"]))
                target_directions.append(np.asarray(sample.action["target_direction"], dtype=np.float32))

            # Convert data to numpy array/matrix
            frame_ids = np.asarray(frame_ids, dtype=np.int32)
            timestamps = np.asarray(timestamps, dtype=np.float64)

            bev_data = np.stack(bev_data_list, axis=0)
            current_locations = np.stack(current_locations, axis=0)
            velocities = np.stack(velocities, axis=0)
            speeds = np.asarray(speeds, dtype=np.float32)
            motion_headings = np.asarray(motion_headings, dtype=np.float32)
            goal_locations = np.stack(goal_locations, axis=0)

            target_speeds = np.asarray(target_speeds, dtype=np.float32)
            target_directions = np.stack(target_directions, axis=0)

            # --- Create HDF5 dataset ---
            # Create groups
            ped_grp = episode_grp.create_group(f"ped_{ped_id}")
            state_grp = ped_grp.create_group("state")
            action_grp = ped_grp.create_group("action")

            # ped_grp.create_dataset("ped_id", data=np.asarray([ped_id], dtype=np.int32))
            ped_grp.create_dataset("frame_id", data=frame_ids)
            ped_grp.create_dataset("timestamp", data=timestamps)

            state_grp.create_dataset("bev_data", data=bev_data, compression="gzip")
            state_grp.create_dataset("current_location", data=current_locations)
            state_grp.create_dataset("velocity", data=velocities)
            state_grp.create_dataset("speed", data=speeds)
            state_grp.create_dataset("motion_heading", data=motion_headings)
            state_grp.create_dataset("goal_location", data=goal_locations)

            action_grp.create_dataset("target_speed", data=target_speeds)
            action_grp.create_dataset("target_direction", data=target_directions)

    print(f"[convert_to_dataset] Saved {len(episode_data)} samples to {output_path}::{group_name}")


class PedestrianStepDataset(Dataset):
    """
    One sample = one pedestrian at one timestep.

    HDF5 structure:
        episode_xxx/
            ped_xxx/
                action/
                    target_direction
                    target_speed
                frame_id
                state/
                    bev_data
                    current_location
                    goal_location
                    motion_heading
                    speed
                    velocity
                timestamp
    """

    def __init__(self, h5_path, use_goal_relative=True):
        self.h5_path = h5_path
        self.use_goal_relative = use_goal_relative
        self.index = []
        self._h5_file = None

        self._build_index()

    def _build_index(self):
        """Build flat index: [(episode_name, ped_name, t), ...]"""
        with h5py.File(self.h5_path, "r") as f:
            for episode_name in f.keys():
                episode_group = f[episode_name]

                for ped_name in episode_group.keys():
                    ped_group = episode_group[ped_name]

                    # number of timesteps for this pedestrian
                    n_steps = ped_group["state"]["bev_data"].shape[0]

                    for t in range(n_steps):
                        self.index.append((episode_name, ped_name, t))

    def _get_h5(self):
        """
        Lazily open HDF5 file.
        Better than reopening the file on every __getitem__ call.
        """
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        f = self._get_h5()
        episode_name, ped_name, t = self.index[idx]

        ped_group = f[episode_name][ped_name]

        # ----- state -----
        bev_data = ped_group["state"]["bev_data"][t]                  # (H, W, C)
        current_location = ped_group["state"]["current_location"][t]  # (2,) or (3,)
        goal_location = ped_group["state"]["goal_location"][t]        # (2,) or (3,)
        motion_heading = ped_group["state"]["motion_heading"][t]      # scalar
        speed = ped_group["state"]["speed"][t]                        # scalar
        velocity = ped_group["state"]["velocity"][t]                  # (2,) or (3,)

        # ----- action -----
        target_speed = ped_group["action"]["target_speed"][t]         # scalar
        target_direction = ped_group["action"]["target_direction"][t] # (2,) or (3,)

        # ----- metadata -----
        frame_id = ped_group["frame_id"][t]
        timestamp = ped_group["timestamp"][t]

        # Convert to numpy float32
        bev_data = np.asarray(bev_data, dtype=np.float32)
        current_location = np.asarray(current_location, dtype=np.float32)
        goal_location = np.asarray(goal_location, dtype=np.float32)
        velocity = np.asarray(velocity, dtype=np.float32)
        target_direction = np.asarray(target_direction, dtype=np.float32)

        t = np.int32(t)
        frame_id = np.int32(frame_id)
        timestamp = np.float32(timestamp)

        motion_heading = np.float32(motion_heading)
        speed = np.float32(speed)
        target_speed = np.float32(target_speed)

        if self.use_goal_relative:
            goal_rel = goal_location - current_location
        else:
            goal_rel = goal_location.copy()

        sample = {
            # inputs
            "bev_data": torch.from_numpy(bev_data),                      # (H, W, C)
            "current_location": torch.from_numpy(current_location),
            "goal_location": torch.from_numpy(goal_location),
            "goal_rel": torch.from_numpy(goal_rel),
            "velocity": torch.from_numpy(velocity),
            "motion_heading": torch.tensor(motion_heading, dtype=torch.float32),
            "speed": torch.tensor(speed, dtype=torch.float32),

            # targets
            "target_speed": torch.tensor(target_speed, dtype=torch.float32),
            "target_direction": torch.from_numpy(target_direction),

            # metadata
            "episode": episode_name,
            "ped_id": ped_name,
            "timestep": t,
            "frame_id": torch.tensor(frame_id),
            "timestamp": torch.tensor(timestamp),
        }

        return sample

    def close(self):
        if getattr(self, "_h5_file", None) is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass
            finally:
                self._h5_file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

def test_dataloader():
    dataset_pth = "./datasets/pedestrian_dataset.h5"
    dataset = PedestrianStepDataset(h5_path=dataset_pth)

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))

    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(f"{k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    test_dataloader()