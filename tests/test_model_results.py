import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.pipeline import Pipeline

from datto.ModelResults import ModelResults

mr = ModelResults()

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
y_train = np.array([1, 0, 0, 0, 1, 0, 0, 1])
X_test = pd.DataFrame(
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
y_test = np.array([1, 0, 0, 0, 1, 0, 0, 1])

X_text = pd.DataFrame(
    [
        ["some text", 1],
        ["some other text", 1],
        ["i like bananas", 2],
        ["i like apples", 2],
        ["running is fun", 3],
        ["jumping is fun running is fun", 3],
        ["do you read mysteries? running is fun", 4],
        ["do you read nonfiction?", 4],
        ["some text", 1],
        ["some other text", 1],
        ["i like bananas", 2],
        ["i like apples", 2],
        ["running is fun", 3],
        ["jumping is fun running is fun", 3],
        ["do you read mysteries? running is fun", 4],
        ["do you read nonfiction?", 4],
        ["some text", 1],
        ["some other text", 1],
        ["i like bananas", 2],
        ["i like apples", 2],
        ["running is fun", 3],
        ["jumping is fun running is fun", 3],
        ["do you read mysteries? running is fun", 4],
        ["do you read nonfiction?", 4],
    ],
    columns=["text", "group_id"],
)


def test_most_similar_texts():
    num_topics = 4
    num_examples = 5
    text_column_name = "text"

    top_words_df, topics_assigned_df = mr.most_similar_texts(
        X_text, num_examples, text_column_name, num_topics
    )

    assert top_words_df.shape[0] == num_topics


def test_coefficients_summary():
    results_df = mr.coefficients_summary(X_train, y_train, 5, 3, "classification")

    assert results_df.shape[0] == 3


def test_most_common_words_by_group():
    results_df = mr.most_common_words_by_group(X_text, "text", "group_id", 3, 1, 3)

    assert X_text["group_id"].nunique() == results_df.shape[0]


def test_score_final_model_classification():
    full_pipeline = Pipeline([("model", LogisticRegression()),])
    fit_pipeline, y_predicted = mr.score_final_model(
        "classification", X_train, y_train, X_test, y_test, full_pipeline
    )

    assert len(y_predicted) == y_test.shape[0]


def test_score_final_model_regression():
    full_pipeline = Pipeline([("model", ElasticNet()),])
    fit_pipeline, y_predicted = mr.score_final_model(
        "regression", X_train, y_train, X_test, y_test, full_pipeline
    )

    assert len(y_predicted) == y_test.shape[0]
