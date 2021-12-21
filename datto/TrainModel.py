import csv
import datetime
import random
from operator import itemgetter

import lightgbm as lgb
import numpy as np
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
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import (
    ElasticNet,
    LogisticRegression,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor


class TrainModel:
    """
    Select & train models
    """

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
        file_name="gridsearching_results",
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

            cleaned_date = (
                datetime.datetime.today().isoformat(" ", "seconds").replace(" ", "-")
            )

            temp_df = pd.DataFrame(lst_dict)
            temp_df["timestamp"] = cleaned_date

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

    def run_feature_selection(self, X_train, y_train, k, is_multiclass):
        """
        Run SelectKBest feature selection for given datasets.
        Implements a custom method of feature selection for multiclass targets.

        Parameters
        --------
        X_train: DataFrame
        y_train: DataFrame
        k: int
        is_multiclass: bool

        Returns
        --------
        cols_to_keep: list

        """
        if is_multiclass:
            all_feature_scores = []
            for label in y_train.columns:
                # Select from all first, limit to k after means
                selector = SelectKBest(chi2, k="all")
                selector.fit(X_train, y_train[label])
                all_feature_scores.append(list(selector.scores_))
            # Get mean feature scoring across all target classes
            mean_feature_scores = np.mean(all_feature_scores, axis=0)
            # Remove nulls
            no_nulls_scores = np.nan_to_num(mean_feature_scores)
            # Sort descending values & keep top k
            selected_feature_idxs = np.argsort(no_nulls_scores)[::-1][:k]
            cols_to_keep = list(itemgetter(*selected_feature_idxs)(X_train.columns))
        else:
            selector = SelectKBest(k=k)
            selector.fit(X_train, y_train)
            cols_to_keep = list(
                X_train.columns[np.where(selector.get_support() == True)].values
            )

        return cols_to_keep

    def train_in_chunks(
        self, X_train, y_train, model_type, is_multiclass, chunk_sizes=500000
    ):
        """
        For large datasets, train model in managable chunk sizes

        Parameters
        --------
        X_train: DataFrame
        y_train: DataFrame
        model_type: str
            'classification' or 'regression'
        is_multiclass: bool
        chunk_sizes: int

        Returns
        --------
        model: sklearn model
        """
        # Set default num_chunks to number needed to get manageable chunk sizes
        # Used this number after testing time needed to train various sizes
        num_chunks = round(y_train.shape[0] / chunk_sizes)
        # Make sure num_chunks isn't below 1
        if num_chunks < 1:
            num_chunks = 1

        step = round(y_train.shape[0] / num_chunks)
        idx_start = 0
        idx_end = step

        if is_multiclass:
            # Need this custom model to do partial_fit with multiclass data
            model = MultiOutputClassifier(MultinomialNB())
        elif model_type == "classification":
            model = SGDClassifier()
        else:
            model = SGDRegressor()

        while idx_start < y_train.shape[0]:
            if is_multiclass:
                model.partial_fit(
                    X_train.iloc[idx_start:idx_end],
                    y_train.iloc[idx_start:idx_end],
                    classes=[
                        np.array(np.unique(y_train[col])) for col in y_train.columns
                    ],
                )
            elif model_type == "classification":
                model = model.partial_fit(
                    X_train.iloc[idx_start:idx_end],
                    y_train.iloc[idx_start:idx_end],
                    classes=np.array(np.unique(y_train)),
                )
            else:
                model = model.partial_fit(
                    X_train.iloc[idx_start:idx_end],
                    y_train.iloc[idx_start:idx_end],
                )

            idx_start += step
            idx_end += step

        return model
