import argparse
import pathlib
import subprocess

from deepsysid.pipeline.gridsearch import ExperimentSessionReport

from utils import load_environment

from relinet.utils import retrieve_tested_models

EXPLAINED_MODEL_BASE_NAMES = [
    'LSTM+Init',
    'ReLiNet',
    'StableReLiNet'
]


def main():
    parser = argparse.ArgumentParser('Explain best-performing models on industrial robot dataset.')
    parser.add_argument('device')
    args = parser.parse_args()

    device_idx = int(args.device)

    main_path = pathlib.Path(__file__).parent.parent.absolute()
    report_path = main_path.joinpath('configuration').joinpath('progress-industrial-robot.json')
    environment_path = main_path.joinpath('environment').joinpath('industrial-robot.env')

    tested_models = retrieve_tested_models(report_path)
    tested_models = set(
        model for model in tested_models
        if model.split('-')[0] in EXPLAINED_MODEL_BASE_NAMES
    )
    environment = load_environment(environment_path)
    for idx, model in enumerate(tested_models):
        return_code = subprocess.call([
            'deepsysid',
            'explain',
            '--mode=test',
            '--enable-cuda',
            f'--device-idx={device_idx}',
            model
        ], env=environment)

        if return_code != 0:
            print(
                f'Failure in running explain on {model}. '
            )

        print(
            f'Explained {idx + 1}/{len(best_models)}.'
        )


if __name__ == '__main__':
    main()
