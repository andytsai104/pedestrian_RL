import json
import math
import os
import random
import weakref

import carla
import cv2
import numpy as np

from ..data_collection.bev.bev_sample import BEVSample, BEVWrapper
from .data_utils import rotate_local_to_world_2d, rotate_world_to_local_2d
from ..models.td3_model import TD3Agent
from .config_loader import load_config
from .sim_utils import (
    AggressiveVehicles,
    CrossroadPedestrians,
    Spector,
    cleanup_simulation,
    refresh_sim,
)


class PedestrianRLEnv:
    '''
    RL environment for one TD3-controlled pedestrian in CARLA.

    Observation matches the BC policy input style:
        bev_data
        velocity_local
        speed
        yaw_sin
        yaw_cos
        goal_rel_local

    Action format:
        [target_speed, dir_right, dir_forward]
    '''

    def __init__(
        self,
        sim_config_name="sim_config.json",
        training_config_name="training_config.json",
        no_rendering_mode=True,
        render_bev=False,
        device="cuda",
    ):
        # ----- load config -----
        self.sim_config = load_config(sim_config_name)
        self.training_config = load_config(training_config_name)

        sim_cfg = self.sim_config["simulation"]
        td3_cfg = self.training_config["td3"]
        td3_params = td3_cfg["params"]

        self.fixed_delta_seconds = sim_cfg["fixed_delta_seconds"]
        self.max_episode_steps = sim_cfg["max_episode_steps"]
        self.warmup_ticks = sim_cfg["warmup_ticks"]
        self.max_ped_speed = sim_cfg["pedestrian"]["speed_range"][1]

        self.goal_scale = float(td3_params["goal_scale"])
        self.num_background_pedestrians = int(td3_params["num_background_pedestrians"])
        self.stall_speed_threshold = float(td3_params["stall_speed_threshold"])
        self.goal_reached_threshold = float(td3_params["goal_reached_threshold"])
        self.clip_bound = float(td3_params["clip_bound"])
        self.render_bev = bool(render_bev)
        self.device = device

        self.reward_weight = td3_cfg["reward"]

        # ----- connect to CARLA -----
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.world_map = self.world.get_map()

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        settings.no_rendering_mode = no_rendering_mode
        self.world.apply_settings(settings)

        # ----- setup scenario -----
        self.intersection_position = carla.Location(
            x=sim_cfg["intersection"]["x"],
            y=sim_cfg["intersection"]["y"],
            z=sim_cfg["intersection"]["z"],
        )
        self.distance = sim_cfg["intersection"]["dist"]

        self.spector = Spector(
            self.world,
            location=self.intersection_position + carla.Location(z=50),
            dist=self.distance,
        )
        self.aggressive_vehicles = AggressiveVehicles(
            self.client,
            self.world,
            location=self.intersection_position,
        )
        self.crossroad_pedestrians = CrossroadPedestrians(
            self.world,
            location=self.intersection_position,
        )
        self.bev_wrapper = BEVWrapper(cfg=None, world=self.world)

        stuck_detection_cfg = sim_cfg["stuck_detection"]
        self.refresh_conditions = {
            "time_out": stuck_detection_cfg["time_out"],
            "start time": self.world.get_snapshot().timestamp.elapsed_seconds,
            "vehicle": {
                "velocity_threshold": stuck_detection_cfg["vehicle"]["velocity_threshold"],
                "stuck_tracker": {},
                "stuck_time_limit": stuck_detection_cfg["vehicle"]["stuck_time_limit"],
                "stuck_count_limit": stuck_detection_cfg["vehicle"]["stuck_count_limit"],
            },
            "pedestrian": {
                "dist": stuck_detection_cfg["pedestrian"]["dist"],
                "min_peds": stuck_detection_cfg["pedestrian"]["min_pedestrians"],
            },
        }

        # ----- reset-able buffers -----
        self.target_ped = None
        self.target_goal = None
        self.prev_location = None
        self.prev_frame_id = None
        self.prev_goal_distance = None
        self.prev_min_vehicle_distance = None
        self.episode_step = 0
        self.last_collision = False
        self.last_bev_sample = None
        self.last_obs = None
        self.collision_sensor = None

    def attach_collision_sensor(self):
        '''Attach one collision sensor to the target pedestrian.'''
        blueprint = self.world.get_blueprint_library().find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            blueprint,
            carla.Transform(),
            attach_to=self.target_ped,
        )

        weak_self = weakref.ref(self)
        self.collision_sensor.listen(
            lambda event: PedestrianRLEnv.on_collision(weak_self, event)
        )

    @staticmethod
    def on_collision(weak_self, event):
        '''Collision callback: only count vehicle collisions.'''
        self = weak_self()
        if self is None:
            return

        if event.other_actor is not None and "vehicle." in event.other_actor.type_id:
            self.last_collision = True

    def destroy_collision_sensor(self):
        '''Destroy collision sensor safely.'''
        if self.collision_sensor is not None and self.collision_sensor.is_alive:
            self.collision_sensor.destroy()
        self.collision_sensor = None

    def close(self):
        '''Close environment resources.'''
        self.destroy_collision_sensor()
        cv2.destroyAllWindows()

    @staticmethod
    def normalize_direction(direction_local, eps=1e-6):
        '''Normalize local-frame 2D direction vectors.'''
        norm = float(np.linalg.norm(direction_local))
        if norm < eps:
            return np.array([0.0, 1.0], dtype=np.float32)
        return (direction_local / norm).astype(np.float32)

    def compute_velocity_speed(self, ped, frame_id):
        '''Compute current location, velocity, and speed.'''
        loc = ped.get_location()
        current_location = np.array([loc.x, loc.y, loc.z], dtype=np.float32)

        if self.prev_location is None or self.prev_frame_id is None:
            velocity = np.zeros(3, dtype=np.float32)
        else:
            passed_frames = frame_id - self.prev_frame_id
            dt = passed_frames * self.fixed_delta_seconds
            if dt > 0:
                velocity = (current_location - self.prev_location) / dt
            else:
                velocity = np.zeros(3, dtype=np.float32)

        speed = float(np.linalg.norm(velocity[:2]))
        return current_location, velocity, speed

    def get_goal_distance(self, current_location):
        '''Get target pedestrian distance to the goal.'''
        if self.target_goal is None:
            return 0.0
        return float(np.linalg.norm(self.target_goal[:2] - current_location[:2]))

    def get_min_vehicle_distance(self, ped_location):
        '''Get minimum distance between target pedestrian and all vehicles.'''
        vehicles = [veh for veh in self.world.get_actors().filter("vehicle.*") if veh.is_alive]
        if len(vehicles) == 0:
            return 999.0

        return float(min(veh.get_location().distance(ped_location) for veh in vehicles))

    def is_on_driving_lane(self, ped_location):
        '''Check if pedestrian is on drivable lane.'''
        wp = self.world_map.get_waypoint(
            ped_location,
            project_to_road=False,
            lane_type=carla.LaneType.Any,
        )
        return bool(wp is not None and wp.lane_type == carla.LaneType.Driving)

    def build_observation(self):
        '''Build one RL observation from current CARLA state.'''
        frame_id = self.world.get_snapshot().timestamp.frame
        self.last_bev_sample = BEVSample(actor=self.target_ped, bev_wrapper=self.bev_wrapper)
        bev_data = self.last_bev_sample.get_bev().astype(np.float32)

        current_location, velocity, speed = self.compute_velocity_speed(self.target_ped, frame_id)
        yaw_heading = math.radians(self.target_ped.get_transform().rotation.yaw)

        goal_rel_world = self.target_goal - current_location
        velocity_local = rotate_world_to_local_2d(velocity[:2], yaw_heading).astype(np.float32)
        goal_rel_local = rotate_world_to_local_2d(goal_rel_world[:2], yaw_heading)
        goal_rel_local = goal_rel_local / max(self.goal_scale, 1e-6)
        goal_rel_local = np.clip(goal_rel_local, -self.clip_bound, self.clip_bound).astype(np.float32)

        obs = {
            "bev_data": bev_data,
            "velocity_local": velocity_local,
            "goal_rel_local": goal_rel_local,
            "yaw_sin": np.float32(math.sin(yaw_heading)),
            "yaw_cos": np.float32(math.cos(yaw_heading)),
            "speed": np.float32(speed),
        }

        debug_state = {
            "frame_id": frame_id,
            "current_location": current_location,
            "velocity": velocity,
            "speed": speed,
            "yaw_heading": yaw_heading,
            "goal_distance": self.get_goal_distance(current_location),
            "min_vehicle_distance": self.get_min_vehicle_distance(self.target_ped.get_location()),
        }
        return obs, debug_state

    def apply_action(self, action):
        '''Apply one RL action to the target pedestrian.'''
        action = np.asarray(action, dtype=np.float32)
        target_speed = float(np.clip(action[0], 0.0, self.max_ped_speed))
        direction_local = self.normalize_direction(action[1:3])

        yaw_heading = math.radians(self.target_ped.get_transform().rotation.yaw)
        direction_world_xy = rotate_local_to_world_2d(direction_local, yaw_heading)

        control = carla.WalkerControl()
        control.speed = target_speed
        control.jump = False
        control.direction = carla.Vector3D(
            x=float(direction_world_xy[0]),
            y=float(direction_world_xy[1]),
            z=0.0,
        )
        self.target_ped.apply_control(control)

        return np.array([target_speed, direction_local[0], direction_local[1]], dtype=np.float32)

    def compute_reward(self, current_location, speed, ped_location):
        '''Compute reward and reward terms for one step.'''
        reward = 0.0
        reward_terms = {}

        goal_distance = self.get_goal_distance(current_location)
        min_vehicle_distance = self.get_min_vehicle_distance(ped_location)
        on_driving_lane = self.is_on_driving_lane(ped_location)

        # Collision reward
        if self.last_collision:
            reward_terms["collision"] = self.reward_weight["collision"]
            reward += reward_terms["collision"]
        else:
            reward_terms["collision"] = 0.0

        # Reward when approaching vehicles
        if self.prev_min_vehicle_distance is None:
            approach_delta = 0.0
        else:
            approach_delta = self.prev_min_vehicle_distance - min_vehicle_distance
        reward_terms["approach_vehicle"] = self.reward_weight["approach_vehicle"] * float(np.clip(approach_delta, -1.0, 1.0))
        reward += reward_terms["approach_vehicle"]

        # Small goal progress reward to prevent meaningless wandering
        if self.prev_goal_distance is None:
            goal_progress = 0.0
        else:
            goal_progress = self.prev_goal_distance - goal_distance
        reward_terms["goal_progress"] = self.reward_weight["goal_progress"] * float(np.clip(goal_progress, -1.0, 1.0))
        reward += reward_terms["goal_progress"]

        # Stall penalty
        if speed < self.stall_speed_threshold:
            reward_terms["stall"] = self.reward_weight["stall"]
            reward += reward_terms["stall"]
        else:
            reward_terms["stall"] = 0.0

        # Small per-step reward / penalty
        reward_terms["living"] = self.reward_weight["living"]
        reward += reward_terms["living"]

        # Drivable-lane related term
        if on_driving_lane:
            reward_terms["lane"] = self.reward_weight["on_driving"]
            reward += reward_terms["lane"]
        else:
            reward_terms["lane"] = self.reward_weight["off_road"]
            reward += reward_terms["lane"]

        # Goal reached bonus
        goal_reached = goal_distance < self.goal_reached_threshold
        if goal_reached:
            reward_terms["goal_reached"] = self.reward_weight["goal_reached"]
            reward += reward_terms["goal_reached"]
        else:
            reward_terms["goal_reached"] = 0.0

        self.prev_goal_distance = goal_distance
        self.prev_min_vehicle_distance = min_vehicle_distance

        extra_state = {
            "goal_distance": goal_distance,
            "min_vehicle_distance": min_vehicle_distance,
            "on_driving_lane": on_driving_lane,
            "goal_reached": goal_reached,
        }
        return reward, reward_terms, extra_state

    def spawn_target_and_background(self):
        '''Spawn one RL-controlled pedestrian and background AI pedestrians.'''
        self.target_ped = None
        self.target_goal = None

        spawn_points = self.crossroad_pedestrians.get_ped_spawn_points(
            self.crossroad_pedestrians.ped_num,
            self.crossroad_pedestrians.in_intersection,
        )
        random.shuffle(spawn_points)

        # ----- spawn target pedestrian first -----
        while len(spawn_points) > 0 and self.target_ped is None:
            spawn_location = spawn_points.pop()
            destination = self.world.get_random_location_from_navigation()

            if destination is None:
                continue
            destination.z += 1.0

            ped = self.crossroad_pedestrians.spawn_single_walker(
                spawn_location=spawn_location,
                destination=destination,
                controller="manual",
                max_speed=self.max_ped_speed,
            )

            if ped is None:
                continue

            self.target_ped = ped
            self.target_goal = np.asarray(
                self.crossroad_pedestrians.ped_goal_loc[ped.id],
                dtype=np.float32,
            )

        if self.target_ped is None:
            raise RuntimeError("Failed to spawn RL-controlled pedestrian.")

        # ----- spawn background pedestrians -----
        max_background = max(self.crossroad_pedestrians.ped_num - 1, 0)
        background_target = min(self.num_background_pedestrians, max_background)

        background_spawned = 0
        while background_spawned < background_target and len(spawn_points) > 0:
            spawn_location = spawn_points.pop()
            destination = self.world.get_random_location_from_navigation()

            if destination is None:
                continue
            destination.z += 1.0

            ped = self.crossroad_pedestrians.spawn_single_walker(
                spawn_location=spawn_location,
                destination=destination,
                controller="ai",
            )

            if ped is not None:
                background_spawned += 1

    def reset(self):
        '''Reset one RL episode.'''
        self.destroy_collision_sensor()
        cleanup_simulation(self.world)
        self.crossroad_pedestrians.reset_pedestrians()

        self.spector.set_spector()
        self.aggressive_vehicles.aggressive_vehicles_spawn()
        self.spawn_target_and_background()

        if self.warmup_ticks > 0:
            for _ in range(self.warmup_ticks):
                self.world.tick()

        self.attach_collision_sensor()
        self.refresh_conditions["start time"] = self.world.get_snapshot().timestamp.elapsed_seconds
        self.refresh_conditions["vehicle"]["stuck_tracker"] = {}

        self.prev_location = None
        self.prev_frame_id = None
        self.prev_goal_distance = None
        self.prev_min_vehicle_distance = None
        self.episode_step = 0
        self.last_collision = False

        obs, debug_state = self.build_observation()
        self.last_obs = obs

        self.prev_location = debug_state["current_location"].copy()
        self.prev_frame_id = debug_state["frame_id"]
        self.prev_goal_distance = debug_state["goal_distance"]
        self.prev_min_vehicle_distance = debug_state["min_vehicle_distance"]

        info = {
            "episode_step": self.episode_step,
            "reset": True,
            "ped_id": self.target_ped.id,
            "goal_distance": debug_state["goal_distance"],
            "min_vehicle_distance": debug_state["min_vehicle_distance"],
        }
        return obs, info

    def step(self, action):
        '''Run one environment step.'''
        applied_action = self.apply_action(action)
        self.world.tick()
        self.episode_step += 1

        sim_state, should_refresh = refresh_sim(
            world=self.world,
            refresh_conditions=self.refresh_conditions,
            intersection_position=self.intersection_position,
        )

        terminated = False
        truncated = False
        term_reason = None

        if self.target_ped is None or not self.target_ped.is_alive:
            truncated = True
            term_reason = "pedestrian_missing"
            info = {
                "episode_step": self.episode_step,
                "ped_id": None,
                "action": applied_action,
                "reward_terms": {},
                "collision": self.last_collision,
                "goal_distance": None,
                "min_vehicle_distance": None,
                "on_driving_lane": None,
                "terminated": terminated,
                "truncated": truncated,
                "term_reason": term_reason,
            }
            return self.last_obs, 0.0, terminated, truncated, info

        obs, debug_state = self.build_observation()
        self.last_obs = obs

        current_location = debug_state["current_location"]
        speed = debug_state["speed"]
        ped_location = self.target_ped.get_location()

        reward, reward_terms, extra_state = self.compute_reward(current_location, speed, ped_location)

        if self.last_collision:
            terminated = True
            term_reason = "vehicle_collision"
        elif extra_state["goal_reached"]:
            terminated = True
            term_reason = "goal_reached"
        elif should_refresh:
            truncated = True
            term_reason = sim_state
        elif self.episode_step >= self.max_episode_steps:
            truncated = True
            term_reason = "max_episode_steps"

        self.prev_location = current_location.copy()
        self.prev_frame_id = debug_state["frame_id"]

        info = {
            "episode_step": self.episode_step,
            "ped_id": self.target_ped.id,
            "action": applied_action,
            "reward_terms": reward_terms,
            "collision": self.last_collision,
            "goal_distance": extra_state["goal_distance"],
            "min_vehicle_distance": extra_state["min_vehicle_distance"],
            "on_driving_lane": extra_state["on_driving_lane"],
            "terminated": terminated,
            "truncated": truncated,
            "term_reason": term_reason,
        }

        if self.render_bev and self.last_bev_sample is not None:
            image = self.last_bev_sample.visualize_bev()
            cv2.imshow("TD3 RL BEV", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

        return obs, float(reward), terminated, truncated, info

    def sample_random_action(self):
        '''Sample one random continuous action.'''
        speed = np.float32(np.random.uniform(0.0, self.max_ped_speed))
        theta = np.random.uniform(-math.pi, math.pi)
        direction = np.array([math.sin(theta), math.cos(theta)], dtype=np.float32)
        return np.array([speed, direction[0], direction[1]], dtype=np.float32)
