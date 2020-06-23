import datetime
import random

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor


class TrainModel:
    def __init__(self):
        self.classifier_param_list = [
            {
                "model": [LogisticRegression(fit_intercept=False)],
                "model__C": [1, 5, 10],
            },
            {
                "model": [DecisionTreeClassifier()],
                "model__min_samples_split": [0.25, 0.5, 1.0],
                "model__max_depth": [5, 10, 15],
            },
            {
                "model": [RandomForestClassifier()],
                "model__min_samples_split": [0.25, 0.5, 1.0],
                "model__max_depth": [5, 10, 15],
            },
            {
                "model": [BaggingClassifier()],
                "model__n_estimators": [5, 10, 15],
                "model__max_features": [0.25, 0.5, 1.0],
            },
            {
                "model": [AdaBoostClassifier()],
                "model__n_estimators": [5, 10, 15],
                "model__learning_rate": [0.001, 0.01, 0.1],
            },
            {
                "model": [MLPClassifier()],
                "model__activation": ["identity", "logistic", "tanh", "relu"],
                "model__alpha": [0.001, 0.01, 0.1],
            },
            {
                "model": [XGBClassifier()],
                "model__n_estimators": [5, 10, 15],
                "model__learning_rate": [0.001, 0.01, 0.1],
            },
        ]

        self.regressor_param_list = [
            {
                "model": [ElasticNet(fit_intercept=False)],
                "model__alpha": [0.001, 0.01, 0.1],
                "model__l1_ratio": [0.25, 0.5, 1.0],
            },
            {
                "model": [DecisionTreeRegressor()],
                "model__min_samples_split": [0.25, 0.5, 1.0],
                "model__max_depth": [5, 10, 15],
            },
            {
                "model": [RandomForestRegressor()],
                "model__min_samples_split": [0.25, 0.5, 1.0],
                "model__max_depth": [5, 10, 15],
            },
            {
                "model": [BaggingRegressor()],
                "model__n_estimators": [5, 10, 15],
                "model__max_features": [0.25, 0.5, 1.0],
            },
            {
                "model": [AdaBoostRegressor()],
                "model__n_estimators": [5, 10, 15],
                "model__learning_rate": [0.001, 0.01, 0.1],
            },
            {
                "model": [MLPRegressor()],
                "model__activation": ["identity", "logistic", "tanh", "relu"],
                "model__alpha": [0.001, 0.01, 0.1],
            },
            {
                "model": [XGBRegressor()],
                "model__n_estimators": [5, 10, 15],
                "model__learning_rate": [0.001, 0.01, 0.1],
            },
        ]

    def train_test_split_by_ids(self, df, id_col, target_col, prop_train):
        """
        Parameters
        --------
        df: DataFrame
        id_col: str
        target_col: str
        prop_train: float

        Returns
        --------
        X_train: DataFrame
        y_train: DataFrame
        X_test: DataFrame
        y_test: DataFrame
        """
        ids = list(set(df[id_col].values))
        random.shuffle(ids)
        len_ids = len(ids)
        number_to_select = int(len_ids * prop_train)

        X_train_ids = pd.DataFrame(ids[:number_to_select], columns=[id_col])
        X_test_ids = pd.DataFrame(ids[number_to_select:], columns=[id_col])

        X_train = pd.merge(df, X_train_ids, how="inner")
        X_test = pd.merge(df, X_test_ids, how="inner")

        y_train = X_train[target_col]
        y_test = X_test[target_col]

        return X_train, y_train, X_test, y_test

    def model_testing(
        self, X_train, y_train, full_pipeline, model_type, tie_breaker_scoring_method,
    ):
        """
        Gridsearches using a Pipeline for best models/params out of a list of commonly used

        Parameters
        --------
        X_train: DataFrame
        y_train: DataFrame
        full_pipeline: sklearn Pipeline
        model_type: str
            'classification' or 'regression'
        tie_breaker_scoring_method: str
            For classification: "precision", "recall", or "roc_auc"
            For regression: "neg_root_mean_squared_error", "neg_median_absolute_error", or "r2"
        
        Returns
        --------
        best_params: dict
        """
        if model_type == "classification":
            lst_scoring_methods = ["precision", "recall", "roc_auc"]
            param_list = self.classifier_param_list
        else:
            lst_scoring_methods = [
                "neg_root_mean_squared_error",
                "neg_median_absolute_error",
                "r2",
            ]
            param_list = self.regressor_param_list

        g = GridSearchCV(
            full_pipeline,
            param_list,
            cv=3,
            n_jobs=-2,
            verbose=2,
            scoring=lst_scoring_methods,
            refit=tie_breaker_scoring_method,
        )
        g.fit(X_train, y_train)

        if model_type == "classification":
            all_scores = list(
                zip(
                    g.cv_results_["params"],
                    g.cv_results_["mean_test_recall"],
                    g.cv_results_["mean_test_precision"],
                    g.cv_results_["mean_test_roc_auc"],
                )
            )

            all_scores.sort(key=lambda x: x[1], reverse=True)
            formatted_scores = [
                (
                    "Params: {}".format(x[0]),
                    "Mean Recall: {0:.4f}".format(x[1]),
                    "Mean Precision: {0:.4f}".format(x[2]),
                    "Mean ROC AUC: {0:.4f}".format(x[3]),
                )
                for x in all_scores
            ]
        else:
            all_scores = list(
                zip(
                    g.cv_results_["params"],
                    g.cv_results_["mean_test_neg_root_mean_squared_error"],
                    g.cv_results_["mean_test_neg_median_absolute_error"],
                    g.cv_results_["mean_test_r2"],
                )
            )

            all_scores.sort(key=lambda x: x[1], reverse=True)
            formatted_scores = [
                (
                    "Params: {}".format(x[0]),
                    "Mean Negative Root Mean Squared Errror: {0:.4f}".format(x[1]),
                    "Mean Negative Median Absolute Error: {0:.4f}".format(x[2]),
                    "Mean R2: {0:.4f}".format(x[3]),
                )
                for x in all_scores
            ]

        # Cleaner printing
        print("\n\n")
        print(
            "*** Best Parameters Using {} | Tie Breaker: {} | {} ***".format(
                lst_scoring_methods,
                tie_breaker_scoring_method,
                datetime.datetime.today().strftime("%Y-%m-%d %H:%m"),
            )
        )
        [
            print("{}\n{}\n{}\n{}\n\n".format(x[0], x[1], x[2], x[3]))
            for x in formatted_scores[:5]
        ]

        return g.best_params_
