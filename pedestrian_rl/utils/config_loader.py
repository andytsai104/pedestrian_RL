import json
import os

def load_config(config_name, config_path="configs/"):
    config_path = os.path.join(config_path, config_name)
    with open(config_path, 'r') as file:
        cfg = json.load(file)
    return cfg
