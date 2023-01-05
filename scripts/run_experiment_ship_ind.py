import argparse
import pathlib
import subprocess

from relinet.utils import load_environment, run_full_gridsearch_session


def main():
    parser = argparse.ArgumentParser('Run experiments for the 4-DOF ship in-distribution dataset.')
    parser.add_argument('device')
    args = parser.parse_args()

    device_idx = int(args.device)

    main_path = pathlib.Path(__file__).parent.parent.absolute()
    report_path = main_path.joinpath('configuration').joinpath('progress-ship.json')
    environment_path = main_path.joinpath('environment').joinpath('ship-ind.env')

    environment = load_environment(environment_path)

    run_full_gridsearch_session(
        report_path=report_path,
        device_idx=device_idx,
        environment=environment
    )


if __name__ == '__main__':
    main()
