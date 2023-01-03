import pathlib
from typing import Set

import h5py
import numpy as np
import pandas as pd
from deepsysid.pipeline.configuration import ExperimentGridSearchTemplate, ExperimentConfiguration
from deepsysid.pipeline.data_io import build_score_file_name, build_explanation_result_file_name
from deepsysid.pipeline.evaluation import ReadableEvaluationScores
from deepsysid.pipeline.gridsearch import ExperimentSessionReport


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


def get_best_models(report_path: pathlib.Path) -> Set[str]:
    report = ExperimentSessionReport.parse_file(report_path)

    if report.best_per_class is None or report.best_per_base_name is None:
        raise ValueError(
            'Did not run "deepsysid session" with action=TEST_BEST yet. '
            'Best performing models have not been identified yet.'
        )

    best_models = set(report.best_per_class.values()).union(report.best_per_base_name.values())

    return best_models


def summarize_prediction_scores(
    configuration: ExperimentConfiguration,
    models: Set[str],
    result_directory: pathlib.Path
) -> pd.DataFrame:
    rows = []
    for model in models:
        score_file_name = build_score_file_name(
            mode='test',
            window_size=configuration.window_size,
            horizon_size=configuration.horizon_size,
            extension='json'
        )
        scores = ReadableEvaluationScores.parse_file(
            result_directory.joinpath(model).joinpath(score_file_name)
        )
        nrmse_single = np.mean(scores.scores_per_horizon[1]['nrmse'])
        nrmse_multi = np.mean(
            scores.scores_per_horizon[configuration.horizon_size]['nrmse']
        )
        rows.append([
            model, nrmse_single, nrmse_multi
        ])

    df = pd.DataFrame(
        data=rows,
        columns=['model', 'H=1', f'H={configuration.horizon_size}']
    )
    return df


def summarize_explanation_scores(
    configuration: ExperimentConfiguration,
    models: Set[str],
    result_directory: pathlib.Path
) -> pd.DataFrame:
    rows = []
    for model in models:
        explanation_file_name = build_explanation_result_file_name(
            mode='test',
            window_size=configuration.window_size,
            horizon_size=configuration.horizon_size,
            extension='hdf5'
        )

        with h5py.File(
            result_directory.joinpath(model).joinpath(explanation_file_name)
        ) as f:
            for metric_name in f.keys():
                for explainer_name in f[metric_name].keys():
                    score = f[metric_name][explainer_name]['score']
                    rows.append([
                        model,
                        metric_name,
                        explainer_name,
                        score
                    ])

    df = pd.DataFrame(
        data=rows,
        columns=['model', 'metric', 'explainer', 'score']
    )
    return df


def summarize_experiment(
    report_path: pathlib.Path,
    environment_path: pathlib.Path
) -> None:
    result_directory = get_results_directory(environment_path)
    best_models = get_best_models(report_path)

    configuration_path = get_configuration_path(environment_path)
    configuration = ExperimentConfiguration.from_grid_search_template(
        ExperimentGridSearchTemplate.parse_file(configuration_path)
    )

    prediction_scores = summarize_prediction_scores(
        configuration,
        best_models,
        result_directory
    )
    explanation_scores = summarize_explanation_scores(
        configuration,
        best_models,
        result_directory
    )

    prediction_scores.to_csv(
        result_directory.joinpath('summary-prediction.csv')
    )
    explanation_scores.to_csv(
        result_directory.joinpath('summary-explanation.csv')
    )


def main():
    main_path = pathlib.Path(__file__).parent.parent.absolute()

    summarize_experiment(
        report_path=main_path.joinpath('configuration').joinpath('progress-ship.json'),
        environment_path=main_path.joinpath('environment').joinpath('ship-ind.env')
    )

    summarize_experiment(
        report_path=main_path.joinpath('configuration').joinpath('progress-ship.json'),
        environment_path=main_path.joinpath('environment').joinpath('ship-ood.env')
    )

    summarize_experiment(
        report_path=main_path.joinpath('configuration').joinpath('progress-industrial-robot.json'),
        environment_path=main_path.joinpath('environment').joinpath('industrial-robot.env')
    )


if __name__ == '__main__':
    main()
