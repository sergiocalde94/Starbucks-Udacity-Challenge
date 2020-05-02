import pytest
import pandas as pd

from sklearn.datasets import load_iris

from starbucks_campaigns_analytics.models import (preprocessing_baseline,
                                                  DataSplitsUnitException,
                                                  DataSplitsSizeException)

TARGET = 'target'

iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris['feature_names'])
df_iris[TARGET] = iris.target


def test_preprocessing_baseline_test_and_valid_unit():
    """Tests if correctly raised data split unit custom exception"""
    with pytest.raises(DataSplitsUnitException):
        preprocessing_baseline(df_iris,
                               cat_features=[],
                               target=TARGET,
                               test_size=1,
                               valid_size=.7)

        preprocessing_baseline(df_iris,
                               cat_features=[],
                               target=TARGET,
                               test_size=.2,
                               valid_size=42)


def test_preprocessing_baseline_test_and_valid_size():
    """Tests if correctly raised data split size custom exception"""
    with pytest.raises(DataSplitsSizeException):
        preprocessing_baseline(df_iris,
                               cat_features=[],
                               target=TARGET,
                               test_size=.6,
                               valid_size=.5)

        preprocessing_baseline(df_iris,
                               cat_features=[],
                               target=TARGET,
                               test_size=.2,
                               valid_size=.9)


def test_preprocessing_baseline_proportions():
    """Tests if correctly raised data split size custom exception"""
    features_both_zero, _ = preprocessing_baseline(df_iris,
                                                   cat_features=[],
                                                   target=TARGET,
                                                   test_size=0,
                                                   valid_size=0)

    assert len(features_both_zero.X_train) == 150
    assert len(features_both_zero.X_test) == 0
    assert len(features_both_zero.X_valid) == 0

    features_both_33, _ = preprocessing_baseline(df_iris,
                                                 cat_features=[],
                                                 target=TARGET,
                                                 test_size=1 / 3,
                                                 valid_size=1 / 3)

    assert len(features_both_33.X_train) == 50
    assert len(features_both_33.X_test) == 50
    assert len(features_both_33.X_valid) == 50

    features_test_33_valid_0, _ = preprocessing_baseline(df_iris,
                                                         cat_features=[],
                                                         target=TARGET,
                                                         test_size=1 / 3,
                                                         valid_size=0)

    assert len(features_test_33_valid_0.X_train) == 100
    assert len(features_test_33_valid_0.X_test) == 50
    assert len(features_test_33_valid_0.X_valid) == 0

    features_valid_33_test_0, _ = preprocessing_baseline(df_iris,
                                                         cat_features=[],
                                                         target=TARGET,
                                                         test_size=0,
                                                         valid_size=2 / 3)

    assert len(features_valid_33_test_0.X_train) == 50
    assert len(features_valid_33_test_0.X_test) == 0
    assert len(features_valid_33_test_0.X_valid) == 100
