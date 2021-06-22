import os
import re
import string
from math import ceil
from operator import itemgetter
from random import randrange

import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, nmf
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import ElasticNet, LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from datto.CleanText import CleanText


class ModelResults:
    def most_similar_texts(
        self, X, num_examples, text_column_name, num_topics=None, chosen_stopwords=set()
    ):
        """
        Uses NMF clustering to create n topics based on adjusted word frequencies

        Parameters
        --------
        X: DataFrame
        num_examples: int
        text_column_name: str
        num_topics: int
            Optional - if none algorithm will determine best number

        Returns
        --------
        topic_words_df: DataFrame
            Top 15 words/phrases per topic
        combined_df: DataFrame
            Original text with topic number assigned to each

        """
        X = X[~X[text_column_name].isna()]
        X = X[X[text_column_name] != ""]
        X = X[X[text_column_name] != " "]
        X = X[X[text_column_name] != "NA"]
        X = X[X[text_column_name] != "n/a"]
        X = X[X[text_column_name] != "N/A"]
        X = X[X[text_column_name] != "na"]

        all_stop_words = (
            set(ENGLISH_STOP_WORDS)
            | set(["-PRON-"])
            | set(string.punctuation)
            | set([" "])
            | chosen_stopwords
        )

        ct = CleanText()
        vectorizer = TfidfVectorizer(
            tokenizer=ct.lematize,
            ngram_range=(1, 3),
            stop_words=all_stop_words,
            min_df=5,
            max_df=0.4,
        )
        vectors = vectorizer.fit_transform(X[text_column_name]).todense()

        # Adding words/phrases used in text data frequencies back into the dataset (so we can see feature importances later)
        vocab = vectorizer.get_feature_names()
        vector_df = pd.DataFrame(vectors, columns=vocab, index=X.index)

        if X.shape[0] < 20:
            return "Too few examples to categorize."

        if not num_topics:

            # In case 1, add 1 to get at least 2
            # The rest are based on eyeballing numbers
            min_topics = ceil(X.shape[0] * 0.01) + 1
            max_topics = ceil(X.shape[0] * 0.2)
            step = ceil((max_topics - min_topics) / 5)

            topic_nums = list(np.arange(min_topics, max_topics, step))

            texts = X[text_column_name].apply(ct.lematize)

            # In gensim a dictionary is a mapping between words and their integer id
            dictionary = Dictionary(texts)

            # Filter out extremes to limit the number of features
            dictionary.filter_extremes(no_below=2, no_above=0.85, keep_n=5000)

            # Create the bag-of-words format (list of (token_id, token_count))
            corpus = [dictionary.doc2bow(text) for text in texts]

            coherence_scores = []

            for num in topic_nums:
                model = nmf.Nmf(
                    corpus=corpus,
                    num_topics=num,
                    id2word=dictionary,
                    chunksize=2000,
                    passes=5,
                    kappa=0.1,
                    minimum_probability=0.01,
                    w_max_iter=300,
                    w_stop_condition=0.0001,
                    h_max_iter=100,
                    h_stop_condition=0.001,
                    eval_every=10,
                    normalize=True,
                    random_state=42,
                )

                cm = CoherenceModel(
                    model=model, texts=texts, dictionary=dictionary, coherence="u_mass"
                )

                coherence_scores.append(round(cm.get_coherence(), 5))

            scores = list(zip(topic_nums, coherence_scores))
            chosen_num_topics = sorted(scores, key=itemgetter(1), reverse=True)[0][0]
        else:
            chosen_num_topics = num_topics

        model = NMF(n_components=chosen_num_topics, random_state=42)
        model.fit(vectors)
        component_loadings = model.transform(vectors)

        top_topics = pd.DataFrame(
            np.argmax(component_loadings, axis=1), columns=["top_topic_num"]
        )

        top_topic_loading = pd.DataFrame(
            np.max(component_loadings, axis=1), columns=["top_topic_loading"]
        )

        X.reset_index(inplace=True, drop=False)
        vector_df.reset_index(inplace=True, drop=True)

        # Fix for duplicate text_column_name
        vector_df.columns = [x + "_vector" for x in vector_df.columns]

        combined_df = pd.concat([X, vector_df, top_topics, top_topic_loading], axis=1)

        combined_df.sort_values(by="top_topic_loading", ascending=False, inplace=True)

        combined_df = pd.concat([X, vector_df, top_topics], axis=1)

        topic_words = {}
        sample_texts_lst = []
        for topic, comp in enumerate(model.components_):
            word_idx = np.argsort(comp)[::-1][:num_examples]
            topic_words[topic] = [vocab[i] for i in word_idx]
            sample_texts_lst.append(
                list(
                    combined_df[combined_df["top_topic_num"] == topic][
                        text_column_name
                    ].values[:num_examples]
                )
            )

        topic_words_df = pd.DataFrame(
            columns=[
                "topic_num",
                "num_in_category",
                "top_words_and_phrases",
                "sample_texts",
            ]
        )

        topic_words_df["topic_num"] = [k for k, _ in topic_words.items()]
        topic_words_df["num_in_category"] = (
            combined_df.groupby("top_topic_num").count().iloc[:, 0]
        )
        topic_words_df["top_words_and_phrases"] = [x for x in topic_words.values()]
        topic_words_df["sample_texts"] = sample_texts_lst

        topic_words_explode = pd.DataFrame(
            topic_words_df["sample_texts"].tolist(), index=topic_words_df.index,
        )

        topic_words_explode.columns = [
            "example{}".format(num) for num in range(len(topic_words_explode.columns))
        ]

        concated_topics = pd.concat(
            [
                topic_words_df[
                    ["topic_num", "num_in_category", "top_words_and_phrases"]
                ],
                topic_words_explode,
            ],
            axis=1,
        )

        print("Topics created with top words & example texts:")
        print(concated_topics)

        original_plus_topics = combined_df[list(X.columns) + ["index", "top_topic_num"]]
        original_with_keywords = pd.merge(
            original_plus_topics,
            concated_topics[["topic_num", "top_words_and_phrases"]],
            left_on="top_topic_num",
            right_on="topic_num",
            how="left",
        ).drop("top_topic_num", axis=1)

        return (
            concated_topics,
            original_with_keywords,
            model,
        )

    def coefficients_graph(self, X_train, X_test, model, model_type, filename):
        """
        Displays graph of feature importances.

        * Number of horizontal axis indicates magnitude of effect on
            target variable (e.g. affected by 0.25)
        * Red/blue indicates feature value (increasing or decreasing feature 
            has _ effect)
        * Blue & red mixed together indicate there isn't a clear 
            effect on the target variable
        * For classification - interpreting magnitude number / x axis - changes the 
            predicted probability of y on average by _ percentage points (axis value * 100)

        Parameters
        --------
        X_train: pd.DataFrame
        X_test: pd.DataFrame
        model: fit model object
        model_type: str
            'classification' or 'regression'
        filename: str
        """
        if not os.path.exists("../images"):
            os.makedirs("../images/")

        if model_type.lower() == "classification":
            f = lambda x: model.predict_proba(x)[:, 1]
        else:
            f = lambda x: model.predict(x)
        med = X_train.median().values.reshape((1, X_train.shape[1]))
        explainer = shap.KernelExplainer(f, med)
        # Runs too slow if X_test is huge, take a representative sample
        if X_test.shape[0] > 1000:
            X_test_sample = X_test.sample(1000)
        else:
            X_test_sample = X_test
        shap_values = explainer.shap_values(X_test_sample)
        shap.summary_plot(shap_values, X_test_sample)
        plt.tight_layout()
        plt.savefig(f"../images/{filename}.png")

        return shap_values

    def most_common_words_by_group(
        self, X, text_col_name, group_col_name, num_examples, num_times_min, min_ngram,
    ):
        """
        Get the most commons phrases for defined groups.

        Parameters
        --------
        X: DataFrame
        text_col_name: str
        group_col_name: str
        num_examples: int
            Number of text examples to include per group
        num_times_min: int
            Minimum number of times word/phrase must appear in texts
        min_ngram: int

        Returns
        --------
        overall_counts_df: DataFrame
            Has groups, top words, and counts

        """
        # Fix for when column name is the same as an ngram column name
        X["group_column"] = X[group_col_name]

        # Remove all other unneeded columns
        X = X[[text_col_name, "group_column"]]

        all_stop_words = (
            set(ENGLISH_STOP_WORDS)
            | set(["-PRON-"])
            | set(string.punctuation)
            | set([" "])
        )

        cv = CountVectorizer(
            stop_words=all_stop_words,
            ngram_range=(min_ngram, 3),
            min_df=num_times_min,
            max_df=0.4,
        )
        vectors = cv.fit_transform(X[text_col_name]).todense()
        words = cv.get_feature_names()
        vectors_df = pd.DataFrame(vectors, columns=words)

        group_plus_vectors = pd.concat([vectors_df, X.reset_index(drop=False)], axis=1)

        count_words = pd.DataFrame(
            group_plus_vectors.groupby("group_column").count()["index"]
        )
        count_words = count_words.loc[:, ~count_words.columns.duplicated()]
        # Fix for when "count" is an ngram column
        count_words.columns = ["count_ngrams"]

        group_plus_vectors = group_plus_vectors.merge(
            count_words, on="group_column", how="left"
        )

        group_plus_vectors["count_ngrams"].fillna(0, inplace=True)

        sums_by_col = (
            group_plus_vectors[
                group_plus_vectors.columns[
                    ~group_plus_vectors.columns.isin([text_col_name, "index",])
                ]
            ]
            .groupby("group_column")
            .sum()
        )

        sums_by_col.sort_values(by="count_ngrams", ascending=False, inplace=True)

        sums_by_col.drop("count_ngrams", axis=1, inplace=True)

        array_sums = np.array(sums_by_col)
        sums_values_descending = -np.sort(-array_sums, axis=1)
        sums_indices_descending = (-array_sums).argsort()

        highest_sum = pd.DataFrame(sums_values_descending[:, 0])
        highest_sum.columns = ["highest_sum"]

        sums_by_col["highest_sum"] = highest_sum["highest_sum"].values

        overall_counts_df = pd.DataFrame(columns=["group_name", "top_words_and_counts"])
        i = 0
        for row in sums_by_col.index:
            dict_scores = {}
            temp_df = pd.DataFrame(columns=["group_name", "top_words_and_counts"])
            temp_df["group_name"] = [row]
            top_columns = sums_by_col.columns[
                sums_indices_descending[i][:num_examples]
            ].values
            top_counts = sums_values_descending[i][:num_examples]
            [dict_scores.update({x: y}) for x, y in zip(top_columns, top_counts)]
            temp_df["top_words_and_counts"] = [dict_scores]
            overall_counts_df = overall_counts_df.append([temp_df])
            print(f"Group Name: {row}\n")
            for k, v in dict_scores.items():
                print(k, v)
            print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
            i += 1

        return overall_counts_df

    def score_final_model(
        self, model_type, X_test, y_test, trained_model, multiclass=False
    ):
        """
        Score your model on the test dataset. Only run this once to get an idea of how your model will perform in realtime.
        Run it after you have chosen your model & parameters to avoid problems with overfitting.

        Parameters
        --------
        model_type: str
        X_test: DataFrame
        y_test: DataFrame
        trained_model: sklearn model
        multiclass: bool

        Returns
        --------
        model: model
            Fit model
        y_predicted: array
        """
        # Predict actual scores
        y_predicted = trained_model.predict(X_test)

        if multiclass:
            pscore = round(precision_score(y_test, y_predicted, average="weighted"), 3)
            rscore = round(recall_score(y_test, y_predicted, average="weighted"), 3)
            f1score = round(f1_score(y_test, y_predicted, average="weighted"), 3)

            print(f"Final Model Precision Weighted: {pscore}")
            print(f"Final Model Recall Weighted: {rscore}")
            print(f"Final Model F1 Weighted: {f1score}")
        elif model_type.lower() == "classification":
            pscore = round(precision_score(y_test, y_predicted), 3)
            rscore = round(recall_score(y_test, y_predicted), 3)
            accuracy = round(accuracy_score(y_test, y_predicted), 3)
            roc_auc = round(roc_auc_score(y_test, y_predicted), 3)

            print(f"Final Model Precision: {pscore}")
            print(f"Final Model Recall: {rscore}")
            print(f"Final Model Accuracy: {accuracy}")
            print(f"Final Model ROC AUC: {roc_auc}")

            crosstab = pd.crosstab(
                y_test, y_predicted, rownames=["Actual"], colnames=["Predicted"],
            )
            print(crosstab)
            sum_crosstab = crosstab.to_numpy().sum()
            print(
                pd.crosstab(
                    y_test, y_predicted, rownames=["Actual"], colnames=["Predicted"],
                ).apply(lambda r: round(r / sum_crosstab, 3))
            )
        else:
            mse = mean_squared_error(y_test, y_predicted)
            mae = median_absolute_error(y_test, y_predicted)
            r2 = round(r2_score(y_test, y_predicted), 3)

            print(
                f"Mean Negative Root Mean Squared Errror: {round((mse ** 5) * -1, 3)}"
            )
            print(f"Mean Negative Median Absolute Error: {round((mae * -1), 3)}")
            print(f"Mean R2: {r2}")

        return trained_model, y_predicted

    def coefficients_summary(
        self, X, y, num_repetitions, num_coefficients, model_type, multiclass=False,
    ):
        """
        Prints average coefficient values using a regression model.

        Parameters
        --------
        X: DataFrame
        y: DataFrame
        num_repetitions: int
            Number of times to create models
        num_coefficients: int
            Number of top coefficients to display
        model_type: str
            'classification' or 'regression'
        multiclass: bool
            
        Returns
        --------
        simplified_df: DataFrame
            Has mean, median, and standard deviation for coefficients after several runs
        """
        coefficients_df = pd.DataFrame(
            columns=["coeff", "pvals", "conf_lower", "conf_higher"]
        )
        X["intercept"] = 1

        for _ in range(num_repetitions):

            X_train, _, y_train, _ = train_test_split(X, y)

            # Fix for Singular matrix error
            vt = VarianceThreshold(0)
            vt.fit(X_train)
            cols_to_keep = X_train.columns[np.where(vt.get_support() == True)].values
            X_train = X_train[cols_to_keep]

            if multiclass:
                model = sm.MNLogit(
                    np.array(y_train.astype(float)), X_train.astype(float)
                )
            elif model_type.lower() == "classification":
                model = sm.Logit(np.array(y_train.astype(float)), X_train.astype(float))
            else:
                model = sm.OLS(np.array(y_train.astype(float)), X_train.astype(float))

            results = model.fit()

            features = results.params.index

            if multiclass:
                pvals = [x[0] for x in results.pvalues.values]
                coeff = [x[0] for x in results.params.values]
                conf_lower = results.conf_int()["lower"].values
                conf_higher = results.conf_int()["upper"].values
            else:
                pvals = results.pvalues.values
                coeff = results.params.values
                conf_lower = results.conf_int()[0]
                conf_higher = results.conf_int()[1]

            temp_df = pd.DataFrame(
                {
                    "features": features,
                    "pvals": pvals,
                    "coeff": coeff,
                    "conf_lower": conf_lower,
                    "conf_higher": conf_higher,
                }
            )
            temp_df = temp_df[
                ["features", "coeff", "pvals", "conf_lower", "conf_higher"]
            ].reset_index(drop=True)

            coefficients_df = coefficients_df.append(temp_df)

        summary_coefficients_df = pd.DataFrame(
            coefficients_df.groupby("features").agg(["mean", "median",])
        ).reset_index(drop=False)

        summary_coefficients_df.columns = [
            "_".join(col) for col in summary_coefficients_df.columns
        ]

        summary_coefficients_df.sort_values("pvals_mean", inplace=True, ascending=True)

        simplified_df = summary_coefficients_df.head(num_coefficients).round(3)

        print("Coefficients summary (descending by mean abs se value):")
        print(simplified_df)

        return simplified_df

    def coefficients_individual_predictions(
        self,
        trained_model,
        X_train,
        X_test,
        id_col,
        num_samples,
        model_type,
        class_names=["False", "True"],
    ):
        if not os.path.exists("../images"):
            os.makedirs("../images/")

        def model_preds_adjusted(data):
            if model_type.lower() == "classification":
                predictions = np.array(trained_model.predict_proba(data))
            else:
                predictions = np.array(trained_model.predict(data))
            return predictions

        if model_type.lower() == "classification":
            explainer = lime.lime_tabular.LimeTabularExplainer(
                np.array(X_train),
                feature_names=X_train.columns,
                class_names=class_names,
                mode="classification",
            )
        else:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                np.array(X_train),
                feature_names=X_train.columns,
                class_names=class_names,
                mode="regression",
            )

        for i in range(num_samples):
            user_idx = randrange(X_test.shape[0])
            exp = explainer.explain_instance(
                np.array(X_test.iloc[user_idx]),
                model_preds_adjusted,
                num_features=50,
                top_labels=10,
            )

            user_id = X_test.iloc[user_idx][id_col]
            print(f"\nUser: {user_id}")

            if model_type.lower() == "classification":
                prediction = class_names[
                    trained_model.predict_proba(
                        pd.DataFrame(X_test.iloc[user_idx]).T
                    ).argmax()
                ]
            else:
                prediction = round(
                    trained_model.predict(pd.DataFrame(X_test.iloc[user_idx]).T)[0], 7
                )

            print('We predicted "{}" for this user because...\n'.format(prediction))
            features_list = []
            for feature in exp.as_list():
                features_list.append({user_id: exp.as_list()})
                try:
                    feature_name = re.findall("<.*<|>.*>", feature[0])[0]
                except Exception:
                    feature_name = re.findall(".*<|.*>", feature[0])[0]
                cleaned_feature_name = (
                    feature_name.replace("<=", "")
                    .replace(">=", "")
                    .replace("<", "")
                    .replace(">", "")
                    .strip()
                )
                comparison_val = float(
                    feature[0].split(" <")[-1].split(" >")[-1].split("=")[-1].strip()
                )
                actual_feature_val = X_test.iloc[user_idx][cleaned_feature_name]
                last_operator = feature[0].split(" ")[-2]

                if "<" in last_operator and actual_feature_val <= comparison_val:
                    print(" * This is true: " + feature[0])
                else:
                    print(" * This is false: " + feature[0])
            exp.as_pyplot_figure()
            plt.tight_layout()
            plt.savefig(f"../images/lime_graph_user_{user_id}.png")

            print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

        return features_list

