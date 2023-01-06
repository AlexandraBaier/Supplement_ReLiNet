import math
import pathlib
from typing import Set

import h5py
import numpy as np
import pandas as pd
from deepsysid.pipeline.configuration import ExperimentGridSearchTemplate, ExperimentConfiguration
from deepsysid.pipeline.data_io import build_score_file_name, build_explanation_result_file_name
from deepsysid.pipeline.evaluation import ReadableEvaluationScores
from deepsysid.pipeline.gridsearch import ExperimentSessionReport


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
    configuration_path: pathlib.Path,
    result_directory: pathlib.Path,
) -> None:
    best_models = get_best_models(report_path)

    configuration = ExperimentConfiguration.from_grid_search_template(
        ExperimentGridSearchTemplate.parse_file(configuration_path)
    )
    n_runs = configuration.session.total_runs_for_best_models

    prediction_scores = summarize_prediction_scores(
        configuration,
        best_models,
        result_directory
    )
    prediction_scores['run'] = 0
    for run_idx in range(1, n_runs):
        additional_prediction_scores = summarize_prediction_scores(
            configuration,
            best_models,
            result_directory=result_directory.joinpath(f'repeat-{run_idx}')
        )
        additional_prediction_scores['run'] = run_idx
        prediction_scores = pd.concat((prediction_scores, additional_prediction_scores))

    # https://stackoverflow.com/a/53522680
    stats = prediction_scores\
        .groupby(['model'])[['H=1', 'H=60']]\
        .agg(['mean', 'count', 'std'])
    horizons = [1, configuration.horizon_size]
    for horizon in horizons:
        mean = stats[(f'H={horizon}', 'mean')]
        count = stats[(f'H={horizon}', 'count')]
        std = stats[(f'H={horizon}', 'std')]
        stats[(f'H={horizon}', 'ci95-width')] = 1.96 * std / np.sqrt(count)
        stats[(f'H={horizon}', 'ci95-lo')] = mean - stats[(f'H={horizon}', 'ci95-width')]
        stats[(f'H={horizon}', 'ci95-hi')] = mean + stats[(f'H={horizon}', 'ci95-width')]

    # explanation_scores = summarize_explanation_scores(
    #     configuration,
    #     best_models,
    #     result_directory
    # )

    prediction_scores.to_csv(
        result_directory.joinpath('summary-prediction.csv'),
    )
    stats.to_csv(
        result_directory.joinpath('summary-prediction-ci.csv'),
    )
    # explanation_scores.to_csv(
    #     result_directory.joinpath('summary-explanation.csv')
    # )


def main():
    main_path = pathlib.Path(__file__).parent.parent.absolute()

    summarize_experiment(
        report_path=main_path.joinpath('configuration').joinpath('progress-ship.json'),
        configuration_path=main_path.joinpath('configuration').joinpath('ship.json'),
        result_directory=main_path.joinpath('results').joinpath('ship-ind')
    )

    summarize_experiment(
        report_path=main_path.joinpath('configuration').joinpath('progress-ship.json'),
        configuration_path=main_path.joinpath('configuration').joinpath('ship.json'),
        result_directory=main_path.joinpath('results').joinpath('ship-ood')
    )

    summarize_experiment(
        report_path=main_path.joinpath('configuration').joinpath('progress-industrial-robot.json'),
        configuration_path=main_path.joinpath('configuration').joinpath('industrial-robot.json'),
        result_directory=main_path.joinpath('results').joinpath('industrial-robot')
    )


if __name__ == '__main__':
    main()
