import carla
import numpy as np

class PedestrianStateAction:
    '''
    Class for storing one pedestrian's state-action sample at a specific simulation frame.

    This class acts as a container for one sampled pedestrian data point. It stores the
    pedestrian ID, the sampled state information, the corresponding action taken by the
    pedestrian controller, and metadata such as frame ID and timestamp.

    Attributes:
        ped_id: ID of the target pedestrian.
        state: Dictionary containing the sampled pedestrian state, including BEV data,
            current location, velocity, speed, motion heading, and goal location.
        action: Dictionary containing the pedestrian action, including target speed
            and target direction.
        state_action_pair: Dictionary that groups pedestrian ID, state, action,
            frame ID, and timestamp into one record.

    Methods:
        set_bev(bev_data):
            Set the BEV observation for the pedestrian state.

        set_location(current_location):
            Set the pedestrian's current 3D location.

        set_velocity(v):
            Set the pedestrian's velocity vector.

        set_speed(speed):
            Set the pedestrian's scalar speed.

        set_heading(motion_heading):
            Set the pedestrian's motion heading in radians.

        set_goal_location(goal_location):
            Set the pedestrian's goal location.

        set_direction(target_direction):
            Set the pedestrian controller's target walking direction.

        set_target_speed(target_speed):
            Set the pedestrian controller's target speed.

        set_actions(target_speed, target_direction):
            Set all action-related fields at once.

        set_states(bev_data, current_location, velocity, speed, motion_heading, goal_location):
            Set all state-related fields at once.
    '''
    
    def __init__(self, target_ped: carla.Walker, frame_id, timestamp):
        self.ped_id = target_ped.id
        self.state = {
            "bev_data": None,               # The (320, 320, N) tensor
            "current_location": None,
            "velocity": None,               # (vx, vy): vx and vy can get heading and speed
            "speed": None,
            "motion_heading": None,         # angle in radians
            "goal_location": None,          # numpy.array (x, y, z) from CrossroadPedestrians
        }
        self.action = {
            "target_speed": None,           # a float number from the controller
            "target_direction": None,       # numpy array (x, y, z) from the controller
        }

        self.state_action_pair = {
            "ped ID": self.ped_id,
            "state": self.state,
            "action": self.action,
            "frame_id": frame_id,
            "timestamp": timestamp
        }

    # State setter
    def set_bev(self, bev_data):
        self.state["bev_data"] = bev_data
    
    def set_location(self, current_location: np.array):
        self.state["current_location"] = current_location
    
    def set_velocity(self, v: np.array):
        self.state["velocity"] = v
    
    def set_speed(self, speed: float):
        self.state["speed"] = speed

    def set_heading(self, motion_heading):
        self.state["motion_heading"] = motion_heading
    
    def set_goal_location(self, goal_location: np.array):
        self.state["goal_location"] = goal_location
    
    # Action setter
    def set_direction(self, target_direction):
        self.action["target_direction"] = target_direction
    
    def set_target_speed(self, target_speed: float):
        self.action["target_speed"] = target_speed

    def set_actions(self, target_speed: float, target_direction):
        self.set_direction(target_direction)
        self.set_target_speed(target_speed)

    def set_states(self, 
                    bev_data,
                    current_location: np.array, 
                    velocity: np.array, 
                    speed: float,
                    motion_heading, 
                    goal_location: np.array
        ):

        self.set_bev(bev_data)
        self.set_location(current_location)
        self.set_velocity(velocity)
        self.set_speed(speed)
        self.set_heading(motion_heading)
        self.set_goal_location(goal_location)

