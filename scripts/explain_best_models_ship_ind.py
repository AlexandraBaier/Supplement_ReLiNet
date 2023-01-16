import argparse
import pathlib
import subprocess

from relinet.utils import load_environment

from relinet.utils import retrieve_tested_models

EXPLAINED_MODELS = [
    #'LSTM+Init-64-3',
    #'ReLiNet-64-2',
    'StableReLiNet-64-3'
]


def main():
    parser = argparse.ArgumentParser('Explain best-performing models on ship in-distribution dataset.')
    parser.add_argument('device')
    args = parser.parse_args()

    device_idx = int(args.device)

    main_path = pathlib.Path(__file__).parent.parent.absolute()
    report_path = main_path.joinpath('configuration').joinpath('progress-ship.json')
    environment_path = main_path.joinpath('environment').joinpath('ship-ind.env')

    tested_models = retrieve_tested_models(report_path)
    tested_models = [
        model for model in tested_models
        if model in EXPLAINED_MODELS
    ]
    environment = load_environment(environment_path)
    for idx, model in enumerate(tested_models):
        return_code = subprocess.call([
            'deepsysid',
            'explain',
            '--enable-cuda',
            '--mode=test',
            f'--device-idx={device_idx}',
            model
        ], env=environment)

        if return_code != 0:
            print(
                f'Failure in running explain on {model}. '
            )

        print(
            f'Explained {idx + 1}/{len(tested_models)}.'
        )


if __name__ == '__main__':
    main()
