# pedestrian_rl/rl/rl_env.py

import carla
import cv2
import math
import random
import numpy as np

from ..utils.config_loader import load_config
from ..utils.sim_utils import (
    Spector,
    CrossroadPedestrians,
    AggressiveVehicles,
    spawn_actors,
    refresh_sim,
    cleanup_simulation
)
from ..data_collection.bev.bev_sample import BEVWrapper, BEVSample


class PedestrianRLEnv:
    '''
    Single-pedestrian RL environment in CARLA.

    This environment is designed for training an RL pedestrian controller using the same observation format as the behavior cloning model.

    Observation:
        - bev_data
        - current_location
        - goal_location
        - goal_rel
        - velocity
        - speed
        - motion_heading

    Action:
        [speed_action, dir_x, dir_y, dir_z]: numpy array of shape (4,)

    Reward:

    '''
    config = load_config("sim_config.json")

    def __init__(self, no_rendering_mode=True, render_bev=False):
        # ----- config -----
        sim_config = self.config["simulation"]

        self.fixed_delta_time = sim_config["fixed_delta_seconds"]
        # self.max_episode_steps = sim_config["max_episode_steps"]
        self.warmup_ticks = sim_config["warmup_ticks"]
        self.max_ped_speed = sim_config["pedestrian"]["speed_range"][1]
        self.render_bev = render_bev

        # ----- CARLA world setup -----
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_time
        settings.no_rendering_mode = no_rendering_mode
        self.world.apply_settings(settings)

        # ----- setup scenario -----
        self.intersection_position = carla.Location(
            x=sim_config["intersection"]["x"],
            y=sim_config["intersection"]["y"],
            z=sim_config["intersection"]["z"],
        )
        self.distance = sim_config["intersection"]["dist"]

        self.spector = Spector(
            self.world,
            location=self.intersection_position + carla.Location(z=50),
            dist=self.distance
        )
        self.aggressive_vehicles = AggressiveVehicles(
            self.client, self.world, location=self.intersection_position
        )
        self.crossroad_pedestrians = CrossroadPedestrians(
            self.world, location=self.intersection_position
        )
        self.bev_wrapper = BEVWrapper(cfg=None, world=self.world)

        # ----- environment states -----
        self.target_ped = None
        self.target_goal = None
        self.prev_ped_location = None
        self.prev_ped_frame = None
        self.episode_step = 0
        self.last_goal_dist = None

        # ----- refresh conditions -----
        stuck_detection_config = sim_config["stuck_detection"]
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

    def reset_episode_tracking(self):
        self.prev_ped_location = None
        self.prev_ped_frame = None
        self.episode_step = 0
        self.last_goal_dist = None

    def find_walker_controller(self, walker):
        '''
        Find the AI controller attached to the walker.
        '''
        controllers = self.world.get_actors().filter("controller.ai.walker")
        for controller in controllers:
            try:
                parent = controller.get_parent()
                if parent is not None and parent.id == walker.id:
                    return controller
            except Exception:
                pass
        return None

    def disable_ai_controller(self, walker):
        '''
        Stop and destroy the original AI controller so RL can fully control the walker.
        '''
        controller = self.find_walker_controller(walker)
        if controller is not None:
            try:
                controller.stop()
            except Exception:
                pass

            try:
                controller.destroy()
                self.world.tick()
            except Exception:
                pass

    def select_target_pedestrian(self):
        '''
        Randomly select one pedestrian in the intersection area.
        '''
        all_peds = list(self.world.get_actors().filter("walker.*"))

        if len(all_peds) == 0:
            return None

        near_peds = []
        for ped in all_peds:
            if ped.get_location().distance(self.intersection_position) < self.distance:
                near_peds.append(ped)

        if len(near_peds) > 0:
            return random.choice(near_peds)

        return random.choice(all_peds)

    def get_velocity_heading(self, ped, frame_id):
        '''
        Compute current velocity, speed, and heading from previous location.
        '''
        current_carla_loc = ped.get_location()
        current_location = np.array(
            [current_carla_loc.x, current_carla_loc.y, current_carla_loc.z],
            dtype=np.float32
        )

        if ped.id is None:
            velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        elif self.prev_ped_location is None:
            velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        else:
            previous_location = self.prev_ped_location
            previous_frame = self.prev_ped_frame
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

        self.prev_ped_location = current_location
        self.prev_ped_frame = frame_id

        return current_location, velocity, speed, motion_heading

    def get_observation(self):
        '''
        Build the RL observation for the target pedestrian.
        '''
        snapshot = self.world.get_snapshot()
        frame_id = snapshot.timestamp.frame

        bev_sample = BEVSample(actor=self.target_ped, bev_wrapper=self.bev_wrapper)
        bev_data = bev_sample.get_bev()

        current_location, velocity, speed, motion_heading = self.get_velocity_heading(
            self.target_ped, frame_id
        )

        goal_location = self.target_goal
        goal_rel = goal_location - current_location

        obs = {
            "bev_data": np.asarray(bev_data, dtype=np.float32),
            "current_location": np.asarray(current_location, dtype=np.float32),
            "goal_location": np.asarray(goal_location, dtype=np.float32),
            "goal_rel": np.asarray(goal_rel, dtype=np.float32),
            "velocity": np.asarray(velocity, dtype=np.float32),
            "speed": np.float32(speed),
            "motion_heading": np.float32(motion_heading),
        }

        return obs, bev_sample

    def apply_action(self, action):
        '''
        Apply RL action to target pedestrian.

        action format:
            action[0] : speed control in [-1, 1]
            action[1] : dir_x
            action[2] : dir_y
            action[3] : dir_z
        '''
        speed_action = float(action[0])
        direction = np.asarray(action[1:4], dtype=np.float32)

        # map speed from [-1, 1] to [0, max_ped_speed]
        target_speed = (speed_action + 1.0) / 2.0
        target_speed = np.clip(target_speed, 0.0, 1.0) * self.max_ped_speed

        # keep motion on ground plane
        direction[2] = 0.0
        direction_norm = np.linalg.norm(direction[:2])

        if direction_norm < 1e-6:
            yaw = math.radians(self.target_ped.get_transform().rotation.yaw)
            direction = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
        else:
            direction = direction / max(np.linalg.norm(direction), 1e-6)

        control = carla.WalkerControl()
        control.speed = float(target_speed)
        control.direction = carla.Vector3D(
            x=float(direction[0]),
            y=float(direction[1]),
            z=float(direction[2]),
        )
        control.jump = False

        self.target_ped.apply_control(control)

        applied_action = {
            "target_speed": target_speed,
            "target_direction": direction
        }

        return applied_action

    def check_collision(self):
        '''
        Simple collision check with nearby vehicles.
        '''
        vehicles = self.world.get_actors().filter("vehicle.*")
        ped_loc = self.target_ped.get_location()

        for veh in vehicles:
            veh_loc = veh.get_location()
            dist = ped_loc.distance(veh_loc)
            if dist < 1.2:
                return True

        return False

    def check_sidewalk(self):
        '''
        Check whether the pedestrian is on sidewalk or shoulder.
        '''
        ped_loc = self.target_ped.get_location()
        world_map = self.world.get_map()

        wp = world_map.get_waypoint(
            ped_loc,
            project_to_road=False,
            lane_type=carla.LaneType.Any
        )

        if wp is None:
            return False

        lane_type = wp.lane_type
        if (lane_type == carla.LaneType.Sidewalk) or (lane_type == carla.LaneType.Shoulder):
            return True

        return False

    def compute_reward(self, obs):
        '''
        Compute reward from current state.
        GOAL: Maximize collission rate + goal direction
        '''
        reward = 0.0

        current_goal_dist = np.linalg.norm(obs["goal_rel"][:2])

        # ----- progress reward -----
        if self.last_goal_dist is not None:
            progress = self.last_goal_dist - current_goal_dist
            reward += 2.0 * progress

        self.last_goal_dist = current_goal_dist

        # ----- time penalty -----
        reward -= 0.5

        # ----- sidewalk penalty -----
        if not self.check_sidewalk():
            reward -= 0.05

        # ----- collision penalty -----
        if self.check_collision():
            reward += 1.0

        return float(reward)

    def check_done(self):
        '''
        Check whether episode should terminate.
        '''
        done = False
        done_reason = None

        if self.target_ped is None:
            done = True
            done_reason = "target_ped_none"

        elif not self.target_ped.is_alive:
            done = True
            done_reason = "target_ped_dead"

        elif self.reached_goal():
            done = True
            done_reason = "goal_reached"

        # elif self.episode_step >= self.max_episode_steps:
        #     done = True
        #     done_reason = "max_episode_steps"

        return done, done_reason

    def reset(self):
        '''
        Reset one RL episode.
        '''
        cleanup_simulation(self.world)
        self.crossroad_pedestrians.reset_pedestrians()

        spawn_actors(
            world=self.world,
            spector=self.spector,
            aggressive_vehicles=self.aggressive_vehicles,
            crossroad_pedestrians=self.crossroad_pedestrians,
        )

        for _ in range(self.warmup_ticks):
            self.world.tick()

        self.target_ped = self.select_target_pedestrian()

        if self.target_ped is None:
            raise RuntimeError("No pedestrian found for RL episode.")

        self.disable_ai_controller(self.target_ped)

        self.target_goal = self.crossroad_pedestrians.ped_goal_loc.get(self.target_ped.id, None)
        if self.target_goal is None:
            loc = self.target_ped.get_location()
            self.target_goal = np.array([loc.x, loc.y, loc.z], dtype=np.float32)
        else:
            self.target_goal = np.asarray(self.target_goal, dtype=np.float32)

        self.reset_episode_tracking()

        self.refresh_conditions["start time"] = self.world.get_snapshot().timestamp.elapsed_seconds
        self.refresh_conditions["vehicle"]["stuck_tracker"] = {}

        obs, bev_sample = self.get_observation()

        info = {
            "ped_id": self.target_ped.id,
            "goal_location": self.target_goal,
        }

        if self.render_bev:
            image = bev_sample.visualize_bev()
            cv2.imshow("RL Env BEV", image)
            cv2.waitKey(1)

        return obs, info

    def step(self, action):
        '''
        One simulation tick with RL action.
        '''
        self.world.tick()

        sim_state, should_refresh = refresh_sim(
            world=self.world,
            refresh_conditions=self.refresh_conditions,
            intersection_position=self.intersection_position
        )

        if should_refresh:
            obs, info = self.reset()
            info["refresh_reason"] = sim_state
            return obs, 0.0, True, info

        applied_action = self.apply_action(action)
        self.episode_step += 1

        obs, bev_sample = self.get_observation()
        reward = self.compute_reward(obs)
        done, done_reason = self.check_done()

        info = {
            "ped_id": self.target_ped.id,
            "episode_step": self.episode_step,
            "done_reason": done_reason,
            "applied_action": applied_action,
        }

        if self.render_bev:
            image = bev_sample.visualize_bev()
            cv2.imshow("RL Env BEV", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                info["keyboard_quit"] = True
                done = True

        return obs, reward, done, info


def test_rl_env():
    env = PedestrianRLEnv(
        no_rendering_mode=False,
        render_bev=True
    )

    obs, info = env.reset()
    print(f"Reset env: ped_id={info['ped_id']}")

    while True:
        action = np.array([
            random.uniform(-1.0, 1.0),   # speed action
            random.uniform(-1.0, 1.0),   # dir x
            random.uniform(-1.0, 1.0),   # dir y
            0.0                          # dir z
        ], dtype=np.float32)

        obs, reward, done, info = env.step(action)

        print(
            f"[RL Step] "
            f"ped_id={info['ped_id']} | "
            f"step={info['episode_step']} | "
            f"reward={reward:.3f} | "
            f"done={done} | "
            f"reason={info['done_reason']}"
        )

        if done:
            print("Episode done. Resetting...")
            obs, info = env.reset()


if __name__ == "__main__":
    try:
        test_rl_env()
    except KeyboardInterrupt:
        print("RL env stopped by user.")
    finally:
        cv2.destroyAllWindows()