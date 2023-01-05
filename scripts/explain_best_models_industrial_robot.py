import argparse
import pathlib
import subprocess

from deepsysid.pipeline.gridsearch import ExperimentSessionReport

from utils import load_environment


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

    report = ExperimentSessionReport.parse_file(report_path)

    if report.best_per_class is None or report.best_per_base_name is None:
        raise ValueError(
            'Did not run "deepsysid session" with action=TEST_BEST yet. '
            'Best performing models have not been identified yet.'
        )

    best_models = set(report.best_per_class.values()).union(report.best_per_base_name.values())
    best_models = set(
        model for model in best_models
        if model.split('-')[0] in EXPLAINED_MODEL_BASE_NAMES
    )
    environment = load_environment(environment_path)
    for idx, model in enumerate(best_models):
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
