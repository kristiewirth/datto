import os

import numpy as np
import pandas as pd
from datto.ModelResults import ModelResults
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

mr = ModelResults()

X_train = pd.DataFrame(
    [
        [1434, 56456, 1],
        [323, 768, 0],
        [5435, 564746456, 0],
        [544, 55567, 21],
        [57978, 58, 2],
        [437, 336, 1],
        [544565, 858, 4],
        [456547, 56456, 10],
    ],
    columns=["id", "webpage", "count"],
)
y_train = np.array([1, 1, 1, 0, 1, 0, 0, 1])
X_test = pd.DataFrame(
    [
        [234, 35656, 1],
        [7878, 435345, 0],
        [3454, 345, 0],
        [78758, 345, 21],
        [234234, 5477, 2],
        [654757, 356536, 1],
        [345, 457457, 4],
        [234, 345, 10],
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
    chosen_num_topics = 4
    num_examples = 5
    text_column_name = "text"
    chosen_stopwords = set(["the"])

    larger_data = pd.concat([X_text, X_text, X_text, X_text], axis=0)

    top_words_df, _, _ = mr.most_similar_texts(
        larger_data,
        num_examples,
        text_column_name,
        chosen_num_topics,
        chosen_stopwords,
        min_df=3,
        max_df=0.4,
        min_ngrams=1,
        max_ngrams=3,
    )

    assert top_words_df.shape[0] == chosen_num_topics


def test_most_similar_texts_no_topic_num():
    chosen_num_topics = None
    num_examples = 5
    text_column_name = "text"
    chosen_stopwords = set(["the"])

    larger_data = pd.concat([X_text, X_text, X_text, X_text], axis=0)

    top_words_df, _, _ = mr.most_similar_texts(
        larger_data,
        num_examples,
        text_column_name,
        chosen_num_topics,
        chosen_stopwords,
        min_df=3,
        max_df=0.4,
        min_ngrams=1,
        max_ngrams=3,
    )

    assert top_words_df.shape[0] <= larger_data.shape[0]


def test_most_common_words_by_group():
    results_df = mr.most_common_words_by_group(X_text, "text", "group_id", 3, 1, 3)

    assert X_text["group_id"].nunique() == results_df.shape[0]


def test_score_final_model_classification():
    model = LogisticRegression(C=0.5, l1_ratio=0.4)
    trained_model = model.fit(X_train, y_train)
    _, y_predicted = mr.score_final_model(
        "classification",
        X_test,
        y_test,
        trained_model,
        csv_file_name="final_model_classification_test",
    )

    assert len(y_predicted) == y_test.shape[0]


def test_score_final_model_regression():
    model = ElasticNet()
    trained_model = model.fit(X_train, y_train)
    _, y_predicted = mr.score_final_model(
        "regression",
        X_test,
        y_test,
        trained_model,
        csv_file_name="final_model_regression_test",
    )

    assert len(y_predicted) == y_test.shape[0]


def test_coefficients_graph_classification():
    model = LogisticRegression()
    model.fit(X_train, y_train)
    shap_values = mr.coefficients_graph(
        X_train, X_test, model, "classification", "classification_test"
    )

    assert isinstance(shap_values, np.ndarray)


def test_coefficients_graph_regression():
    model = ElasticNet()
    model.fit(X_train, y_train)
    shap_values = mr.coefficients_graph(
        X_train, X_test, model, "regression", "regression_test"
    )

    assert isinstance(shap_values, np.ndarray)


def test_coefficients_individual_predictions_classification():
    model = LogisticRegression()
    trained_model = model.fit(X_train, y_train)
    id_col = "id"
    num_id_examples = 3
    num_feature_examples = 5
    model_type = "classification"
    class_names = ["False", "True"]
    features_list = mr.coefficients_individual_predictions(
        trained_model,
        X_train,
        X_train,
        X_test,
        id_col,
        num_id_examples,
        num_feature_examples,
        model_type,
        class_names,
    )
    assert isinstance(features_list, list)


def test_coefficients_individual_predictions_regression():
    model = ElasticNet()
    trained_model = model.fit(X_train, y_train)
    id_col = "id"
    num_id_examples = 3
    num_feature_examples = 5
    model_type = "regression"
    features_list = mr.coefficients_individual_predictions(
        trained_model,
        X_train,
        X_train,
        X_test,
        id_col,
        num_id_examples,
        num_feature_examples,
        model_type,
    )
    assert isinstance(features_list, list)


def test_score_final_model_multiclass():
    y_train = pd.DataFrame(
        [
            [1, 1],
            [1, 0],
            [1, 1],
            [1, 1],
            [1, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ],
        columns=["var1", "var2"],
    )
    y_test = pd.DataFrame(
        [
            [1, 1],
            [1, 0],
            [1, 1],
            [1, 1],
            [1, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ],
        columns=["var1", "var2"],
    )
    model = DecisionTreeClassifier()
    trained_model = model.fit(X_train, y_train)
    _, y_predicted = mr.score_final_model(
        "classification",
        X_test,
        y_test,
        trained_model,
        multiclass=True,
        csv_file_name="final_model_multiclass_test",
    )

    assert len(y_predicted) == y_test.shape[0]


def test_coefficients_summary_multiclass():
    y_train = pd.DataFrame(
        [
            [1, 1],
            [1, 0],
            [1, 1],
            [1, 1],
            [1, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ],
        columns=["var1", "var2"],
    )
    # For some reason this errors if using more than one feature
    results_df = mr.coefficients_summary(
        pd.DataFrame(X_train["count"]), y_train, 5, 3, "classification", multiclass=True
    )
    assert results_df.shape[0] == 1


def test_coefficients_summary_classification():
    results_df = mr.coefficients_summary(
        pd.DataFrame(X_train["count"]), y_train, 5, 3, "classification"
    )
    assert results_df.shape[0] == 1


def test_coefficients_summary_regression():
    results_df = mr.coefficients_summary(
        pd.DataFrame(X_train["count"]), y_train, 5, 3, "regression"
    )
    assert results_df.shape[0] == 1


def test_get_tree_diagram():
    path = "../images/"
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    mr.get_tree_diagram(model, X_train, path)

    assert os.path.exists(f"{path}decision-tree.png")


def test_coefficients_graph_multiclass():
    model = DecisionTreeClassifier()

    y_train = pd.DataFrame(
        [
            [1, 1],
            [1, 0],
            [1, 1],
            [1, 1],
            [1, 0],
            [1, 1],
            [0, 1],
            [1, 1],
        ],
        columns=["var1", "var2"],
    )

    model.fit(X_train, y_train)
    shap_values = mr.coefficients_graph(
        X_train,
        X_test,
        model,
        "classification",
        "multiclass_test",
        multiclass=True,
        y_test=y_train,
    )

    assert isinstance(shap_values, np.ndarray)
