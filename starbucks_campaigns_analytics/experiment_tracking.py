import mlflow
import mlflow.sklearn
import pandas as pd

from collections import namedtuple
from typing import List, Union
from mlflow.exceptions import MlflowException
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline

from .constants import PATH_MLFLOW_TRACKING, PATH_PROJECT


Tracking = namedtuple("Tracking",
                      "run_name tags params metrics model model_name")


def new_experiment(name: str) -> str:
    """Creates an experiment in `PATH_MLFLOW_TRACKING` and
    return the experiment id

    If the experiment already exists, return its id
    """
    mlflow.set_tracking_uri(str(PATH_MLFLOW_TRACKING))
    try:
        return mlflow.create_experiment(name)
    except MlflowException:
        return mlflow.get_experiment_by_name(name).experiment_id


def new_run(experiment_id,
            run_name,
            tags,
            params,
            metrics,
            model,
            model_name) -> str:
    """Creates a new run in the experiment with id `experiment_id`
    and return the run id
    """
    with mlflow.start_run(experiment_id=experiment_id,
                          run_name=run_name) as run:
        mlflow.set_tags(tags)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, model_name)

        return run.info.run_uuid


def apply_runs_to_experiment(experiment_id: str,
                             trackings: List[Tracking]) -> List[str]:
    """Giving an `experiment_id` and a list of trackings,
    which defines each run of the experiment
    """
    return [new_run(experiment_id,
                    run_name=tracking.run_name,
                    tags=tracking.tags,
                    params=tracking.params,
                    metrics=tracking.metrics,
                    model=tracking.model,
                    model_name=tracking.model_name)
            for tracking in trackings]


def n_best_models_from_experiments(experiment_ids: List[str],
                                   n: int,
                                   order_by: List[str]) -> pd.DataFrame:
    """Gets `n` best models from every runs in `experiments_ids`
    """
    return mlflow.search_runs(experiment_ids, max_results=n, order_by=order_by)


def load_trained_model(model_artifact_uri: str,
                       model_model_name: str) -> Union[CatBoostClassifier,
                                                       Pipeline]:
    """Load trained model from `model_artifact_uri`
    giving also its name `model_model_name` uri"""
    return (mlflow
            .sklearn
            .load_model(f'{model_artifact_uri}/{model_model_name}'))


def load_best_model() -> Union[CatBoostClassifier, Pipeline]:
    """Shortcut to get the best model applying
    `load_trained_model`"""
    return load_trained_model(model_artifact_uri=PATH_PROJECT,
                              model_model_name='best_model')
