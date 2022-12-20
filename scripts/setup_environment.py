import dataclasses
import pathlib
import subprocess


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

    dirs.models.joinpath('pelican').mkdir(exist_ok=True)
    dirs.results.joinpath('pelican').mkdir(exist_ok=True)

    return dirs


def create_environment(dirs: Directories) -> None:
    ship_ind_file = dirs.environment.joinpath('ship-ind.env')
    ship_ood_file = dirs.environment.joinpath('ship-ood.env')
    pelican_file = dirs.environment.joinpath('pelican.env')

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
            f'DATASET_DIRECTORY={dirs.datasets.joinpath("pelican")}',
            f'MODELS_DIRECTORY={dirs.models.joinpath("pelican")}',
            f'RESULT_DIRECTORY={dirs.results.joinpath("pelican")}',
            f'CONFIGURATION={dirs.configuration.joinpath("pelican.json")}'
        ]))


def download_datasets(dirs: Directories) -> None:
    subprocess.call([
        'deepsysid',
        'download',
        '4dof-sim-ship',
        dirs.datasets.joinpath('ship-ind'),
        dirs.datasets.joinpath('ship-ood')
    ])
    subprocess.call([
        'deepsysid',
        'download',
        'pelican',
        '--train_fraction=0.6',
        '--validation_fraction=0.1',
        dirs.datasets.joinpath('pelican')
    ])


def main():
    main_path = pathlib.Path(__file__).parent.parent.absolute()
    dirs = create_directories(main_path)
    create_environment(dirs)
    download_datasets(dirs)


if __name__ == '__main__':
    main()
