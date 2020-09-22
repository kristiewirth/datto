import pandas as pd
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.pipeline import Pipeline

from datto.TrainModel import TrainModel

tm = TrainModel()

X_train = pd.DataFrame(
    [
        [1434, 56456, 1],
        [323, 768, 0],
        [5435, 564746456, 1],
        [544, 564456556, 1],
        [54345345, 58, 1],
        [54456565, 336, 1],
        [544565, 858, 1],
        [54365856, 56456, 1],
    ],
    columns=["id", "webpage", "count"],
)
y_train = pd.DataFrame([[1], [0], [0], [0], [1], [0], [0], [1]], columns=["converted"],)


def test_train_test_split_by_ids():
    df = pd.DataFrame(
        [[1434, "page1", 1], [323, "page2", 0], [5435, "page3", 1]],
        columns=["id", "webpage", "converted"],
    )

    X_train, X_test, _, _ = tm.train_test_split_by_ids(df, "id", "converted", 0.7)

    assert X_train.shape[0] > X_test.shape[0]


def test_model_testing_classification():
    full_pipeline = Pipeline([("model", LogisticRegression()),])
    model_type = "classification"
    tie_breaker_scoring_method = "precision"

    best_params = tm.model_testing(
        X_train, y_train, full_pipeline, model_type, tie_breaker_scoring_method,
    )

    assert isinstance(best_params, dict)


def test_model_testing_regression():
    full_pipeline = Pipeline([("model", ElasticNet()),])
    model_type = "regression"
    tie_breaker_scoring_method = "r2"

    best_params = tm.model_testing(
        X_train, y_train, full_pipeline, model_type, tie_breaker_scoring_method,
    )

    assert isinstance(best_params, dict)
