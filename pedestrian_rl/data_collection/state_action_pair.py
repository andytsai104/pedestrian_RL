class PedestrianStateAction:
    '''
    The "Memory" of your model: stores what the pedestrian saw vs. what they did.
    '''
    def __init__(self, actor_id):
        self.actor_id = actor_id
        self.state = {
            "bev": None,         # The (320, 320, N) tensor
            "velocity": None,    # (vx, vy): vx and vy can get heading and speed
            # "dist_to_car": None, # (N, 1)
            "frame_id": None,
            "timestamp": None,   # (t, time_stamp)
        }
        self.action = {
            "target_vel": None,  # The vx, vy the model chose (vx, vy)
            "frame_id": None,
            "timestamp": None,   # (t, time_stamp)
        }