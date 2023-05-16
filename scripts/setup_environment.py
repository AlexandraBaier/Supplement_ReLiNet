import dataclasses
import os
import pathlib
import subprocess
from distutils.dir_util import copy_tree


@dataclasses.dataclass
class Directories:
    environment: pathlib.Path
    configuration: pathlib.Path
    datasets: pathlib.Path
    models: pathlib.Path
    results: pathlib.Path


def create_directories(main_path: pathlib.Path) -> Directories:
    dirs = Directories(
        environment=main_path.joinpath('environment'),
        configuration=main_path.joinpath('configuration'),
        datasets=main_path.joinpath('datasets'),
        models=main_path.joinpath('models'),
        results=main_path.joinpath('results')
    )

    dirs.environment.mkdir(exist_ok=True)
    dirs.datasets.mkdir(exist_ok=True)
    dirs.models.mkdir(exist_ok=True)
    dirs.results.mkdir(exist_ok=True)

    dirs.models.joinpath('ship').mkdir(exist_ok=True)

    dirs.results.joinpath('ship-ind').mkdir(exist_ok=True)
    dirs.results.joinpath('ship-ood').mkdir(exist_ok=True)

    dirs.models.joinpath('industrial-robot').mkdir(exist_ok=True)
    dirs.results.joinpath('industrial-robot').mkdir(exist_ok=True)

    return dirs


def create_environment(dirs: Directories) -> None:
    ship_ind_file = dirs.environment.joinpath('ship-ind.env')
    ship_ood_file = dirs.environment.joinpath('ship-ood.env')
    pelican_file = dirs.environment.joinpath('industrial-robot.env')

    with ship_ind_file.open(mode='w') as f:
        f.write('\n'.join([
            f'DATASET_DIRECTORY={dirs.datasets.joinpath("ship-ind")}',
            f'MODELS_DIRECTORY={dirs.models.joinpath("ship")}',
            f'RESULT_DIRECTORY={dirs.results.joinpath("ship-ind")}',
            f'CONFIGURATION={dirs.configuration.joinpath("ship.json")}'
        ]))

    with ship_ood_file.open(mode='w') as f:
        f.write('\n'.join([
            f'DATASET_DIRECTORY={dirs.datasets.joinpath("ship-ood")}',
            f'MODELS_DIRECTORY={dirs.models.joinpath("ship")}',
            f'RESULT_DIRECTORY={dirs.results.joinpath("ship-ood")}',
            f'CONFIGURATION={dirs.configuration.joinpath("ship.json")}'
        ]))

    with pelican_file.open(mode='w') as f:
        f.write('\n'.join([
            f'DATASET_DIRECTORY={dirs.datasets.joinpath("industrial-robot")}',
            f'MODELS_DIRECTORY={dirs.models.joinpath("industrial-robot")}',
            f'RESULT_DIRECTORY={dirs.results.joinpath("industrial-robot")}',
            f'CONFIGURATION={dirs.configuration.joinpath("industrial-robot.json")}'
        ]))


def download_datasets(dirs: Directories) -> None:
    subprocess.call([
        'deepsysid',
        'download',
        '4dof-sim-ship',
        dirs.datasets.joinpath('ship-ind'),
        dirs.datasets.joinpath('ship-ood')
    ])

    # ship-ood is for testing only, so it does not have a train dataset.
    # However, for evaluation of explainers a train dataset is needed.
    # We solve this by copying over the ship-ind train dataset.
    ship_ind_train_directory = dirs.datasets.joinpath('ship-ind').joinpath('processed').joinpath('train')
    ship_ood_train_directory = dirs.datasets.joinpath('ship-ood').joinpath('processed').joinpath('train')
    os.makedirs(ship_ood_train_directory, exist_ok=True)
    copy_tree(
        str(ship_ind_train_directory),
        str(ship_ood_train_directory)
    )

    subprocess.call([
        'deepsysid',
        'download',
        'industrial-robot',
        # train-test split is 90-10
        # we want a 70-20-10 split
        # this is 18% of the training set for validation
        '--validation_fraction=0.18',
        dirs.datasets.joinpath('industrial-robot')
    ])


def main():
    main_path = pathlib.Path(__file__).parent.parent.absolute()
    dirs = create_directories(main_path)
    create_environment(dirs)
    download_datasets(dirs)


if __name__ == '__main__':
    main()
