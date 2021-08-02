import csv
import datetime
import random

import lightgbm as lgb
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
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
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor


class TrainModel:
    def __init__(self):
        self.classifier_param_list = [
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
                "model": [MLPClassifier()],
                "model__activation": ["identity", "logistic", "tanh", "relu"],
                "model__alpha": [0.001, 0.01, 0.1],
            },
            {
                "model": [LogisticRegression(fit_intercept=False)],
                "model__C": [1, 5, 10],
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
                "model": [XGBClassifier()],
                "model__n_estimators": [5, 10, 15],
                "model__learning_rate": [0.001, 0.01, 0.1],
            },
            {
                "model": [lgb.LGBMClassifier()],
                "model__learning_rate": [0.01, 0.001],
            },
            {
                "model": [CatBoostClassifier()],
                "model__learning_rate": [0.01, 0.001],
            },
        ]

        self.regressor_param_list = [
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
                "model": [MLPRegressor()],
                "model__activation": ["identity", "logistic", "tanh", "relu"],
                "model__alpha": [0.001, 0.01, 0.1],
            },
            {
                "model": [ElasticNet(fit_intercept=False)],
                "model__alpha": [0.001, 0.01, 0.1],
                "model__l1_ratio": [0.25, 0.5, 1.0],
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
                "model": [XGBRegressor()],
                "model__n_estimators": [5, 10, 15],
                "model__learning_rate": [0.001, 0.01, 0.1],
            },
            {
                "model": [lgb.LGBMRegressor()],
                "model__learning_rate": [0.01, 0.001],
            },
            {
                "model": [CatBoostRegressor()],
                "model__learning_rate": [0.01, 0.001],
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

        return X_train, X_test, y_train, y_test

    def model_testing(
        self,
        X_train,
        y_train,
        model_type,
        tie_breaker_scoring_method,
        save_to_csv=True,
        file_name="model_results",
        multiclass=False,
    ):
        """
        Gridsearches using a model for best models/params out of a list of commonly used

        Parameters
        --------
        X_train: DataFrame
        y_train: DataFrame
        model: sklearn model
        model_type: str
            'classification' or 'regression'
        tie_breaker_scoring_method: str
            For classification: "precision", "recall", or "f1"
            For regression: "neg_root_mean_squared_error", "neg_median_absolute_error", or "r2"
        save_to_csv: bool
        file_name: str
        multiclass: bool

        Returns
        --------
        best_params: dict
        """
        if model_type == "classification":
            model = Pipeline(
                [
                    ("model", LogisticRegression()),
                ]
            )
            # Only some models/scoring work with multiclass
            if multiclass:
                param_list = self.classifier_param_list[:3]
                lst_scoring_methods = [
                    "recall_weighted",
                    "precision_weighted",
                    "f1_weighted",
                ]
            else:
                param_list = self.classifier_param_list[:3]
                lst_scoring_methods = ["recall", "precision", "f1"]
        else:
            model = Pipeline(
                [
                    ("model", ElasticNet()),
                ]
            )
            lst_scoring_methods = [
                "neg_root_mean_squared_error",
                "neg_median_absolute_error",
                "r2",
            ]
            param_list = self.regressor_param_list

        g = GridSearchCV(
            model,
            param_list,
            cv=3,
            n_jobs=-2,
            verbose=2,
            scoring=lst_scoring_methods,
            refit=tie_breaker_scoring_method,
        )
        g.fit(X_train, y_train)

        if model_type == "classification":
            if multiclass:
                all_scores = list(
                    zip(
                        g.cv_results_["params"],
                        g.cv_results_["mean_test_recall_weighted"],
                        g.cv_results_["mean_test_precision_weighted"],
                        g.cv_results_["mean_test_f1_weighted"],
                    )
                )

                all_scores.sort(key=lambda x: x[1], reverse=True)
                formatted_scores = [
                    (
                        "Params: {}".format(x[0]),
                        "Mean Recall Weighted: {0:.4f}".format(x[1]),
                        "Mean Precision Weighted: {0:.4f}".format(x[2]),
                        "Mean F1 Weighted: {0:.4f}".format(x[3]),
                    )
                    for x in all_scores
                ]

            else:
                all_scores = list(
                    zip(
                        g.cv_results_["params"],
                        g.cv_results_["mean_test_recall"],
                        g.cv_results_["mean_test_precision"],
                        g.cv_results_["mean_test_f1"],
                    )
                )

                all_scores.sort(key=lambda x: x[1], reverse=True)
                formatted_scores = [
                    (
                        "Params: {}".format(x[0]),
                        "Mean Recall: {0:.4f}".format(x[1]),
                        "Mean Precision: {0:.4f}".format(x[2]),
                        "Mean F1 Score: {0:.4f}".format(x[3]),
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
            for x in formatted_scores[:30]
        ]

        if save_to_csv:
            lst_dict = []
            for model in all_scores[:30]:
                d = dict()
                for k, v in zip(
                    list(model[0].keys()) + lst_scoring_methods,
                    list(model[0].values()) + [x for x in model[1:]],
                ):
                    d[k] = v
                lst_dict.append(d)

            dateTimeObj = datetime.datetime.now()
            timestampStr = dateTimeObj.strftime("%m-%d-%Y (%H:%M:%S.%f)")

            temp_df = pd.DataFrame(lst_dict)
            temp_df["timestamp"] = timestampStr

            try:
                previous_df = pd.read_csv(f"{file_name}.csv")
                model_results_df = pd.concat([previous_df, temp_df], axis=0)
                model_results_df.reset_index(inplace=True, drop=True)
            except Exception:
                model_results_df = temp_df

            model_results_df = model_results_df.reindex(
                sorted(model_results_df.columns), axis=1
            )

            with open(f"{file_name}.csv", "w") as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=",")
                csvwriter.writerow(model_results_df.columns)
                for _, row in model_results_df.iterrows():
                    csvwriter.writerow(row)

        return g.best_params_
