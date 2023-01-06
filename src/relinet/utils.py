import os
import pathlib
import subprocess
from typing import Dict, List, Set

from deepsysid.pipeline.gridsearch import ExperimentSessionReport


def load_environment(environment_path: pathlib.Path) -> Dict[str, str]:
    env = os.environ.copy()
    with environment_path.open(mode='r') as f:
        for line in f:
            var_name, var_value = line.strip().split('=')
            env[var_name] = var_value
    return env


def get_value_from_environment_file(
    environment_path: pathlib.Path,
    environment_variable: str
) -> pathlib.Path:
    with environment_path.open(mode='r') as f:
        for line in f:
            var_name, var_value = line.strip().split('=')
            if var_name == environment_variable:
                return pathlib.Path(var_value).expanduser().absolute()


def get_results_directory(
    environment_file_path: pathlib.Path
) -> pathlib.Path:
    return get_value_from_environment_file(
        environment_file_path,
        'RESULT_DIRECTORY'
    )


def get_configuration_path(
    environment_file_path: pathlib.Path
) -> pathlib.Path:
    return get_value_from_environment_file(
        environment_file_path,
        'CONFIGURATION'
    )


def run_full_gridsearch_session(
    report_path: pathlib.Path,
    device_idx: int,
    environment: Dict[str, str]
):
    if report_path.exists():
        print('Continuing session...')
        action = 'CONTINUE'
        return_code = subprocess.call([
            'deepsysid',
            'session',
            '--enable-cuda',
            f'--device-idx={device_idx}',
            f'--reportin={report_path}',
            report_path,
            action
        ], env=environment)
    else:
        print('Starting session from fresh.')
        action = 'NEW'
        return_code = subprocess.call([
            'deepsysid',
            'session',
            '--enable-cuda',
            f'--device-idx={device_idx}',
            report_path,
            action
        ], env=environment)

    if return_code != 0:
        raise ValueError('Failed running gridsearch session.')

    action = 'TEST_BEST'
    subprocess.call([
        'deepsysid',
        'session',
        '--enable-cuda',
        f'--device-idx={device_idx}',
        f'--reportin={report_path}',
        report_path,
        action
    ], env=environment)


def retrieve_tested_models(report_path: pathlib.Path) -> Set[str]:
    report = ExperimentSessionReport.parse_file(report_path)

    if report.best_per_class is None or report.best_per_base_name is None:
        raise ValueError(
            'Did not run "deepsysid session" with action=TEST_BEST yet. '
            'Best performing models have not been identified yet.'
        )

    return report.tested_models
