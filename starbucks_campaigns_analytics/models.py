import pandas as pd

from collections import namedtuple
from typing import List, Tuple, Union
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from joblib import dump, load

from .constants import TARGET, PATH_MODELS, RANDOM_STATE

Features = namedtuple('Features', 'X_train X_test X_valid')
Labels = namedtuple('Labels', 'y_train y_test y_valid')
Metrics = namedtuple('Metrics', 'ACC AUC')


class DataSplitsUnitException(Exception):
    """Custom exception to make
    `sklearn.model_selection.train_test_split`
    only works with the float unit
    """
    pass


class DataSplitsSizeException(Exception):
    """Customized exception just to clarify "ValueError"
    from `sklearn.model_selection.train_test_split`
    behavior when the test or validation size is not correct
    and none is equal to zero
    """
    pass


def add_past_events_info(df: pd.DataFrame) -> pd.DataFrame:
    """Create dummy variables for `event` column and
    adds shifted features to `df_to_group`

    Be careful, returned dataframe is sorted by time
    """
    df_to_group_sorted = df.sort_values(by=['person', 'time'])
    df_to_group_dummies = pd.get_dummies(df_to_group_sorted,
                                         columns=['event'],
                                         drop_first=True)

    columns_events = [column
                     for column in df_to_group_dummies.columns
                     if column.startswith('event')]

    columns_shifted = (['amount_shifted', f'{TARGET}_shifted']
                       + [f'{column}_shifted' for column in columns_events])

    df_to_group_dummies[columns_shifted] = (
        df_to_group_dummies[['person', 'amount', TARGET] + columns_events]
        .groupby('person')
        .transform('cumsum')
        .shift(1)
    )

    return df_to_group_dummies.fillna(0)


def cat_features_fillna(df: pd.DataFrame,
                        cat_features: List[str]) -> pd.DataFrame:
    """Fills NA values for each column in `cat_features` for
    `df` dataframe
    """
    df_copy = df.copy()

    for cat in cat_features:
        try:
            df_copy[cat] = (
                df_copy[cat].cat.add_categories('UNKNOWN').fillna('UNKNOWN')
            )

        except AttributeError:
            # The dtype is object instead of category
            df_copy[cat] = df_copy[cat].fillna('UNKNOWN')

    return df_copy


def preprocessing_baseline(df: pd.DataFrame,
                           cat_features: List[str],
                           target: str,
                           test_size: float = .15,
                           valid_size: float = .15) -> Tuple[Features, Labels]:
    """Creates `features` and `labels` splits and fill NA values
    for categorical features passed in `cat_features` from data
    in `df` dataframe

    Target feature must be provided in `target` arg

    `test_size` and `valid_size` has to be greater than zero and
    less too one, if it is 0 removes that split set
    """
    if 0 < test_size >= 1 or 0 < valid_size >= 1:
        raise DataSplitsUnitException(
            'The parameters test_size and valid_size have to be '
            'greater than zero and less too one'
        )

    X = df.drop(columns=target)
    y = df[target]

    X_filled = cat_features_fillna(X, cat_features=cat_features)

    try:
        X_train, X_test_and_valid, y_train, y_test_and_valid = (
            train_test_split(
                X_filled,
                y,
                test_size=test_size + valid_size,
                random_state=RANDOM_STATE,
                stratify=y
            )
        )

        X_test, X_valid, y_test, y_valid = (
            train_test_split(X_test_and_valid,
                             y_test_and_valid,
                             test_size=valid_size / (test_size + valid_size),
                             random_state=RANDOM_STATE,
                             stratify=y_test_and_valid)
        )
    except ValueError as value_error:
        if (test_size + valid_size) >= 1:
            raise DataSplitsSizeException(
                'The size of the test and validation data added together '
                'is greater than or equal to one'
            ) from value_error
        elif test_size == valid_size == 0:
            X_train, y_train = X_filled.copy(), y.copy()
            X_test, y_test = pd.DataFrame(), pd.Series(dtype=y.dtype)
            X_valid, y_valid = pd.DataFrame(), pd.Series(dtype=y.dtype)
        elif test_size == 0:
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_filled,
                y,
                test_size=valid_size,
                random_state=RANDOM_STATE,
                stratify=y
            )

            X_test, y_test = pd.DataFrame(), pd.Series(dtype=y.dtype)
        elif valid_size == 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X_filled,
                y,
                test_size=test_size,
                random_state=RANDOM_STATE,
                stratify=y
            )

            X_valid, y_valid = pd.DataFrame(), pd.Series(dtype=y.dtype)
        else:
            raise value_error

    return (Features(X_train, X_test, X_valid),
            Labels(y_train, y_test, y_valid))


def compute_metrics(model: Union[Pipeline, CatBoostClassifier],
                    X: pd.DataFrame,
                    y: pd.Series) -> Metrics:
    """Computes `model` metrics for `X` and
    `y`
    """
    predict = model.predict(X)
    predict_proba = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, predict)
    auc = roc_auc_score(y, predict_proba)

    return Metrics(ACC=acc, AUC=auc)


def show_metrics_baseline(model: Union[Pipeline, CatBoostClassifier],
                          features: Features,
                          labels: Labels) -> None:
    """Giving `model`, `features` and `labels` show accuracy and AUC
    for training, testing and validation data

    Model passed in argument `model` has to be already fitted
    """
    split_names = [field.replace('X_', '').capitalize()
                   for field in features._fields]

    for split_name, split_features, split_labels in zip(split_names,
                                                        features,
                                                        labels):
        if split_features.empty:
            continue

        split_acc, split_auc = compute_metrics(model,
                                               X=split_features,
                                               y=split_labels)

        print(f'Accuracy {split_name}: {split_acc}')
        print(f'AUC {split_name}: {split_auc}')
