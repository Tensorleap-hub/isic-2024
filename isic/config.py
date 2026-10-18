import os
import yaml


def load_config():
    with open(os.path.join(os.path.dirname(__file__), 'project_config.yaml')) as f:
        return yaml.safe_load(f)


CONFIG = load_config()
