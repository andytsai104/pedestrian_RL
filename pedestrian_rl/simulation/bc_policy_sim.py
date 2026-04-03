import os
import math
import argparse
import random
from typing import Optional
import carla
import cv2
import numpy as np
import torch
from ..utils.config_loader import load_config
from ..utils.sim_utils import Spector, CrossroadPedestrians, AggressiveVehicles, spawn_actors, refresh_sim, cleanup_simulation
from ..data_collection.bev.bev_sample import BEVWrapper, BEVSample
from ..data_collection.utils import rotate_world_to_local_2d, rotate_local_to_world_2d
from ..models.bc_model import BehaviorCloningPolicy
from ..models.cnn_encoder import CNNEncoder


class BCPolicyRunner:
    """
    Run a trained Behavior Cloning pedestrian policy in CARLA for one target pedestrian.

    Pipeline per tick:
        CARLA world -> build observation -> BC model -> WalkerControl -> apply_control
    """

    def __init__(
        self,
        checkpoint_path,
        training_config_name="training_config.json",
        sim_config_name="sim_config.json",
        no_rendering_mode=False,
        device="cuda",
        goal_scale=16.0,
    ):
        # --- load config ---
        self.sim_config = load_config(sim_config_name)
        self.training_config = load_config(training_config_name) if training_config_name else None

        sim_cfg = self.sim_config["simulation"]
        self.fixed_delta_seconds = sim_cfg["fixed_delta_seconds"]
        self.max_episode_steps = sim_cfg["max_episode_steps"]
        self.warmup_ticks = sim_cfg["warmup_ticks"]
        self.max_ped_speed = sim_cfg["pedestrian"]["speed_range"][1]
        self.goal_scale = float(goal_scale)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # --- connecting CARLA ---
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        settings.no_rendering_mode = no_rendering_mode
        self.world.apply_settings(settings)

        self.intersection_position = carla.Location(
            x=sim_cfg["intersection"]["x"],
            y=sim_cfg["intersection"]["y"],
            z=sim_cfg["intersection"]["z"],
        )
        self.distance = sim_cfg["intersection"]["dist"]

        # --- intersection sim setup ---
        self.spector = Spector(
            self.world,
            location=self.intersection_position + carla.Location(z=50),
            dist=self.distance,
        )
        self.aggressive_vehicles = AggressiveVehicles(self.client, self.world, location=self.intersection_position)
        self.crossroad_pedestrians = CrossroadPedestrians(self.world, location=self.intersection_position)
        self.bev_wrapper = BEVWrapper(cfg=None, world=self.world)

        # --- build up prediction model from optimal checkpoint ---
        self.model = self._build_model(checkpoint_path)
        self.model.eval()

        # --- initialize buffers
        self.target_ped = None
        self.target_goal = None
        self.prev_location = None
        self.prev_frame = None
        self.current_step = 0

        # --- get stuck conditions from config ---
        stuck_detection_config = sim_cfg["stuck_detection"]
        self.refresh_conditions = {
            "time_out": stuck_detection_config["time_out"],
            "start time": self.world.get_snapshot().timestamp.elapsed_seconds,
            "vehicle": {
                "velocity_threshold": stuck_detection_config["vehicle"]["velocity_threshold"],
                "stuck_tracker": {},
                "stuck_time_limit": stuck_detection_config["vehicle"]["stuck_time_limit"],
                "stuck_count_limit": stuck_detection_config["vehicle"]["stuck_count_limit"],
            },
            "pedestrian": {
                "dist": stuck_detection_config["pedestrian"]["dist"],
                "min_peds": stuck_detection_config["pedestrian"]["min_pedestrians"],
            },
        }

    def _build_model(self, checkpoint_path: str):
        '''
        Build up prediction model from optimal checkpoints
        '''
        bev_feature_dim = 128
        hidden_dim = 256

        if self.training_config is not None:
            bev_feature_dim = self.training_config["cnn"]["bev_feature_dim"]
            hidden_dim = self.training_config["cnn"]["hidden_dim"]

        cnn_encoder = CNNEncoder(input_channels=4, feature_dim=bev_feature_dim)
        model = BehaviorCloningPolicy(
            cnn_encoder=cnn_encoder,
            bev_feature_dim=bev_feature_dim,
            scalar_feature_dim=7,
            hidden_dim=hidden_dim,
            direction_dim=2,
            dropout=0.0,
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        print(f"[BCPolicyRunner] Loaded checkpoint: {checkpoint_path}")
        return model

    def _disable_ai_controller(self, walker: carla.Actor):
        success = self.crossroad_pedestrians.destroy_ped_controller(walker.id)
        self.world.tick()
        if success:
            print(f"[BCPolicyRunner] Destroyed AI controller for ped {walker.id}")

    def _select_target_pedestrian(self):
        walkers = list(self.world.get_actors().filter("walker.*"))
        if len(walkers) == 0:
            return None

        nearby_walkers = [
            ped for ped in walkers
            if ped.get_location().distance(self.intersection_position) < self.distance
        ]
        pool = nearby_walkers if len(nearby_walkers) > 0 else walkers
        return random.choice(pool)

    def _reset_target_tracking(self):
        self.prev_location = None
        self.prev_frame = None
        self.current_step = 0

    def reset_episode(self):
        cleanup_simulation(self.world)
        self.crossroad_pedestrians.reset_pedestrians()

        spawn_actors(
            world=self.world,
            spector=self.spector,
            aggressive_vehicles=self.aggressive_vehicles,
            crossroad_pedestrians=self.crossroad_pedestrians,
        )

        # Warmup simulation
        for _ in range(self.warmup_ticks):
            self.world.tick()

        self.target_ped = self._select_target_pedestrian()
        if self.target_ped is None:
            raise RuntimeError("No pedestrian found after spawning.")

        self._disable_ai_controller(self.target_ped)

        goal = self.crossroad_pedestrians.ped_goal_loc.get(self.target_ped.id, None)
        if goal is None:
            loc = self.target_ped.get_location()
            goal = np.array([loc.x, loc.y, loc.z], dtype=np.float32)
        self.target_goal = np.asarray(goal, dtype=np.float32)

        self._reset_target_tracking()
        self.refresh_conditions["start time"] = self.world.get_snapshot().timestamp.elapsed_seconds
        self.refresh_conditions["vehicle"]["stuck_tracker"] = {}

        print(f"[BCPolicyRunner] New target pedestrian: {self.target_ped.id}")
        print(f"[BCPolicyRunner] Goal location: {self.target_goal}")

    def _compute_velocity_speed(self, ped: carla.Actor, frame_id: int):
        current_loc = ped.get_location()
        current_location = np.array([current_loc.x, current_loc.y, current_loc.z], dtype=np.float32)

        if self.prev_location is None or self.prev_frame is None:
            velocity = np.zeros(3, dtype=np.float32)
        else:
            passed_frames = frame_id - self.prev_frame
            dt = passed_frames * self.fixed_delta_seconds
            if dt > 0:
                velocity = (current_location - self.prev_location) / dt
            else:
                velocity = np.zeros(3, dtype=np.float32)

        speed = float(np.linalg.norm(velocity[:2]))
        self.prev_location = current_location.copy()
        self.prev_frame = frame_id
        return current_location, velocity, speed

    def _build_observation(self, ped: carla.Actor, frame_id: int):
        '''
        Build up observation for target pedestrian
        '''
        bev_sample = BEVSample(actor=ped, bev_wrapper=self.bev_wrapper)
        bev_data = bev_sample.get_bev()

        current_location, velocity, speed = self._compute_velocity_speed(ped, frame_id)
        yaw_heading = math.radians(ped.get_transform().rotation.yaw)

        goal_rel_world = self.target_goal - current_location
        velocity_local = rotate_world_to_local_2d(velocity[:2], yaw_heading)
        goal_rel_local = rotate_world_to_local_2d(goal_rel_world[:2], yaw_heading)
        goal_rel_local = goal_rel_local / max(self.goal_scale, 1e-6)
        goal_rel_local = np.clip(goal_rel_local, -2.0, 2.0).astype(np.float32)

        batch = {
            "bev_data": torch.from_numpy(np.asarray(bev_data, dtype=np.float32)).unsqueeze(0).to(self.device),
            "velocity_local": torch.from_numpy(velocity_local).unsqueeze(0).to(self.device),
            "goal_rel_local": torch.from_numpy(goal_rel_local).unsqueeze(0).to(self.device),
            "yaw_sin": torch.tensor([math.sin(yaw_heading)], dtype=torch.float32, device=self.device),
            "yaw_cos": torch.tensor([math.cos(yaw_heading)], dtype=torch.float32, device=self.device),
            "speed": torch.tensor([speed], dtype=torch.float32, device=self.device),
        }

        debug_state = {
            "bev_sample": bev_sample,
            "current_location": current_location,
            "velocity": velocity,
            "velocity_local": velocity_local,
            "speed": speed,
            "yaw_heading": yaw_heading,
            "goal_rel_local": goal_rel_local,
        }
        return batch, debug_state

    def _postprocess_action(self, outputs, ped: carla.Actor):
        '''
        Process predicted actions to executable actions for controller (local-to-word conversion)
        '''

        pred_speed = float(outputs["pred_speed"][0, 0].item())
        pred_direction_local = outputs["pred_direction"][0].detach().cpu().numpy().astype(np.float32)

        direction_norm = float(np.linalg.norm(pred_direction_local))
        if direction_norm < 1e-6:
            pred_direction_local = np.array([0.0, 1.0], dtype=np.float32)
        else:
            pred_direction_local = pred_direction_local / direction_norm

        yaw_heading = math.radians(ped.get_transform().rotation.yaw)
        pred_direction_world_xy = rotate_local_to_world_2d(pred_direction_local, yaw_heading)

        direction_world = np.array(
            [pred_direction_world_xy[0], pred_direction_world_xy[1], 0.0],
            dtype=np.float32,
        )

        pred_speed = max(0.0, min(pred_speed, float(self.max_ped_speed)))
        return pred_speed, pred_direction_local, direction_world

    def _apply_control(self, ped: carla.Actor, speed: float, direction: np.ndarray, jump: bool = False):
        control = carla.WalkerControl()
        control.speed = float(speed)
        control.jump = bool(jump)
        control.direction = carla.Vector3D(
            x=float(direction[0]),
            y=float(direction[1]),
            z=float(direction[2]),
        )
        ped.apply_control(control)

    def _target_reached(self, ped: carla.Actor, threshold: float = 1.5):
        if self.target_goal is None:
            return False
        current_loc = ped.get_location()
        goal_loc = carla.Location(
            x=float(self.target_goal[0]),
            y=float(self.target_goal[1]),
            z=float(self.target_goal[2]),
        )
        return current_loc.distance(goal_loc) < threshold

    def step_once(self, render_bev=True):
        self.world.tick()

        sim_state, should_refresh = refresh_sim(
            world=self.world,
            refresh_conditions=self.refresh_conditions,
            intersection_position=self.intersection_position,
        )

        if should_refresh:
            print(f"[BCPolicyRunner] Refresh triggered: {sim_state}")
            self.reset_episode()
            return

        if self.target_ped is None or (not self.target_ped.is_alive):
            print("[BCPolicyRunner] Target pedestrian lost. Resetting episode.")
            self.reset_episode()
            return

        snapshot = self.world.get_snapshot()
        frame_id = snapshot.timestamp.frame
        batch, debug_state = self._build_observation(self.target_ped, frame_id)

        with torch.no_grad():
            outputs = self.model(batch)

        pred_speed, pred_direction_local, pred_direction_world = self._postprocess_action(outputs, self.target_ped)
        self._apply_control(self.target_ped, pred_speed, pred_direction_world, jump=False)
        self.current_step += 1

        print(
            f"[BC] ped_id={self.target_ped.id} | step={self.current_step} | frame={frame_id}\n"
            f"  speed(state)         : {debug_state['speed']:.3f}\n"
            f"  velocity_local       : {np.round(debug_state['velocity_local'], 3)}\n"
            f"  pred_speed           : {pred_speed:.3f}\n"
            f"  pred_dir_local       : {np.round(pred_direction_local, 3)}\n"
            f"  goal_rel_local       : {np.round(debug_state['goal_rel_local'], 3)}\n"
        )

        if render_bev:
            image = debug_state["bev_sample"].visualize_bev()
            cv2.imshow("BC Policy Runner BEV", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt

        if self._target_reached(self.target_ped):
            print("[BCPolicyRunner] Target reached. Resetting episode.")
            self.reset_episode()
            return

        if self.current_step >= self.max_episode_steps:
            print("[BCPolicyRunner] Max episode steps reached. Resetting episode.")
            self.reset_episode()
            return

    def run(self, render_bev=True):
        self.reset_episode()
        while True:
            self.step_once(render_bev=render_bev)


def main():
    config_name = "training_config.json"
    config = load_config(config_name=config_name)
    checkpoint_name = "best_model.pt"
    checkpoint_seed_dir = "seed_5"
    checkpoint_path = os.path.join(config["checkpoint_dir"], checkpoint_seed_dir, checkpoint_name)

    runner = BCPolicyRunner(checkpoint_path=checkpoint_path)

    try:
        runner.run(render_bev=True)
    except KeyboardInterrupt:
        print("\n[BCPolicyRunner] Stopped by user.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()