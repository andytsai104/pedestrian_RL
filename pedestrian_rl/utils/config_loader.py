import json


def load_config(config_name):
    config_path = "configs/" + config_name
    with open(config_path, 'r') as file:
        cfg = json.load(file)
    return cfg
