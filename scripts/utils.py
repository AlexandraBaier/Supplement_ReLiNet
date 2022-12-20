import os
import pathlib
from typing import Dict


def load_environment(environment_path: pathlib.Path) -> Dict[str, str]:
    env = os.environ.copy()
    with environment_path.open(mode='r') as f:
        for line in f:
            var_name, var_value = line.strip().split('=')
            env[var_name] = var_value
    return env
