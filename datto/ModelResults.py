import csv
import datetime
import os
import random
import re
import string
import warnings
from html import unescape

import graphviz
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import spacy
import statsmodels.api as sm
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, nmf
from nltk.corpus import stopwords
from sklearn import set_config
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_squared_error,
    median_absolute_error,
    multilabel_confusion_matrix,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from spacy.cli import download
from tabulate import tabulate

from datto.CleanDataframe import CleanDataframe
from datto.utils import *

# Hide precision/recall/f1 warnings for model scorings
warnings.filterwarnings("always")

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


class ModelResults:
    """
    Evaluate model performance & explore output
    """

    def _jaccard_similarity(self, topic_1, topic_2):
        """
        Derives the Jaccard similarity of two topics

        Jaccard similarity:
        - A statistic used for comparing the similarity and diversity of sample sets
        - J(A,B) = (A ∩ B)/(A ∪ B)
        - Goal is low Jaccard scores for coverage of the diverse elements

        Parameters
        --------
        topic_1: list
        topic_2: list


        Returns
        --------
        score: float
        """

        intersection = set(topic_1).intersection(set(topic_2))
        union = set(topic_1).union(set(topic_2))
        score = float(len(intersection)) / float(len(union))

        return score

    def most_similar_texts(
        self,
        X,
        text_column_name,
        chosen_num_topics=None,
        chosen_stopwords=set(),
        exclude_numbers=False,
        exclude_times=False,
        exclude_months=False,
        exclude_weekdays=False,
        exclude_greetings_goodbyes=False,
        exclude_adverbs=False,
        num_examples=15,
        min_df=3,
        max_df=0.1,
        min_ngrams=1,
        max_ngrams=3,
    ):
        """
        Uses NMF clustering to create n topics based on adjusted word frequencies

        Parameters
        --------
        X: DataFrame
        num_examples: int
        text_column_name: str
        chosen_num_topics: int
            Optional - if none algorithm will determine best number
        chosen_stopwords: set
            Option to add in your own unique stopwords
        exclude_numbers: bool
            Adding numbers 0-3000 (with & without commas) as additional stopwords
        exclude_times: bool
            Adding times as additional stopwords (e.g. `8:00`)
        exclude_months: bool
            Adding month names as additional stopwords
        exclude_weekdays: bool
            Adding weekday names as additional stopwords
        exclude_greetings_goodbyes: bool
            Adding common greetings & goodbyes as additional stopwords (e.g. `hello`)
        exclude_adverbs: bool
            Adding common adverbs as additional stopwords (e.g. `especially`)
        min_df: float
            Minimum number/proportion of docs that need to have the words
        max_df: float
            Maximum number/proportion of docs that can have the words
        min_ngrams: int
            Minimum number of words needed in phrases found
        max_ngrams: int
            Maximum number of words in phrases found

        Returns
        --------
        all_topics: DataFrame
            Top n words/phrases per topic
        original_with_keywords: DataFrame
            Original text with topic number assigned to each
        model: NMF model
        """
        X = X[~X[text_column_name].isna()]
        X = X[X[text_column_name] != ""]
        X = X[X[text_column_name] != " "]

        # Remove HTML & unicode characters
        X[text_column_name] = X[text_column_name].apply(
            lambda x: unescape(x)
            .encode("ascii", errors="ignore")
            .decode("ascii")
            .replace("    ", " ")
            .replace("   ", " ")
            .replace("  ", " ")
            .strip()
        )

        with_stopword_params = chosen_stopwords
        if exclude_numbers:
            with_stopword_params = with_stopword_params | numbers
        if exclude_times:
            with_stopword_params = with_stopword_params | times
        if exclude_months:
            with_stopword_params = with_stopword_params | months
        if exclude_weekdays:
            with_stopword_params = with_stopword_params | weekdays
        if exclude_greetings_goodbyes:
            with_stopword_params = with_stopword_params | greetings_goodbyes
        if exclude_adverbs:
            with_stopword_params = with_stopword_params | adverbs

        all_stop_words = (
            # Combine stopwords from all the packages
            set(ENGLISH_STOP_WORDS)
            | set(stopwords.words("english"))
            | nlp.Defaults.stop_words
            | set(string.punctuation)
            | set(string.ascii_lowercase)
            | set(
                [
                    "-PRON-",
                    " ",
                    "i.e",
                    "e.g.",
                    "-",
                    "--",
                    "---",
                    "----",
                    "..",
                    "...",
                    "....",
                    "w/",
                    "^^",
                    "^^^",
                    "^^^^",
                    "’",
                    "~~",
                    "~~~",
                    "~~~~",
                    "and/or",
                ]
            )
            | set(with_stopword_params)
        )

        ct = CleanDataframe()
        vectorizer = TfidfVectorizer(
            tokenizer=ct.lematize,
            ngram_range=(min_ngrams, max_ngrams),
            stop_words=all_stop_words,
            min_df=min_df,
            max_df=max_df,
        )
        vectors = vectorizer.fit_transform(X[text_column_name]).todense()

        # Adding words/phrases used in text data frequencies back into the dataset (so we can see feature importances later)
        vocab = list(vectorizer.get_feature_names_out())
        vector_df = pd.DataFrame(vectors, columns=vocab, index=X.index)

        if not chosen_num_topics:
            # Inspired by: https://bit.ly/3CqH2Zw

            if X.shape[0] <= 5:
                return "Too few examples to categorize."
            elif X.shape[0] <= 75:
                topic_nums = list(np.arange(5, X.shape[0], 5))
            else:
                topic_nums = list(np.arange(5, 75 + 1, 5))

            texts = X[text_column_name].apply(ct.lematize)

            # In gensim a dictionary is a mapping between words and their integer id
            dictionary = Dictionary(texts)

            # Filter out extremes to limit the number of features
            dictionary.filter_extremes(no_below=2, no_above=0.85, keep_n=5000)

            # Create the bag-of-words format (list of (token_id, token_count))
            corpus = [dictionary.doc2bow(text) for text in texts]

            NMF_models = {}
            NMF_topics = {}

            # Iterate through possible topic numbers and make models for each
            for i in topic_nums:
                NMF_models[i] = nmf.Nmf(
                    corpus=corpus,
                    id2word=dictionary,
                    num_topics=i,
                    chunksize=2000,
                    passes=20,
                    random_state=42,
                )

                shown_topics = NMF_models[i].show_topics(
                    num_topics=i, num_words=15, formatted=False
                )
                NMF_topics[i] = [
                    [word[0] for word in topic[1]] for topic in shown_topics
                ]

            # Measure Jaccard similarity between topics
            # Verify that these scores are low, i.e. topics are distinct
            NMF_stability = {}
            for i in range(0, len(topic_nums) - 1):
                jaccard_sims = []
                for _, topic1 in enumerate(NMF_topics[topic_nums[i]]):
                    sims = []
                    for _, topic2 in enumerate(NMF_topics[topic_nums[i + 1]]):
                        sims.append(self._jaccard_similarity(topic1, topic2))

                    jaccard_sims.append(sims)

                NMF_stability[topic_nums[i]] = jaccard_sims

            mean_stabilities = [
                np.array(NMF_stability[i]).mean() for i in topic_nums[:-1]
            ]

            # Measure coherence scores
            # Verify these are high, i.e. important words in topics appear consistently together in the texts
            coherences = [
                CoherenceModel(
                    model=NMF_models[i],
                    texts=texts,
                    dictionary=dictionary,
                    coherence="c_v",
                ).get_coherence()
                for i in topic_nums[:-1]
            ]

            coh_sta_diffs = [
                coherences[i] - mean_stabilities[i]
                for i in range(len(topic_nums[:-1]))[:-1]
            ]
            coh_sta_max = max(coh_sta_diffs)
            coh_sta_max_idxs = [
                i for i, j in enumerate(coh_sta_diffs) if j == coh_sta_max
            ]
            ideal_topic_num_index = coh_sta_max_idxs[0]
            chosen_num_topics = topic_nums[ideal_topic_num_index]

        model = NMF(n_components=chosen_num_topics, random_state=42)
        model.fit(vectors)
        component_loadings = model.transform(vectors)

        top_topics = pd.DataFrame(
            np.argmax(component_loadings, axis=1), columns=["top_topic_num"]
        )
        top_topics["second_topic"] = (-component_loadings).argsort(axis=1)[:, 1]
        top_topics["third_topic"] = (-component_loadings).argsort(axis=1)[:, 2]

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
            examples_lst = [
                x
                for x in list(
                    combined_df[combined_df["top_topic_num"] == topic][
                        text_column_name
                    ].values
                )
                if topic_words[topic][0] in x
                or topic_words[topic][1] in x
                or topic_words[topic][2] in x
                or topic_words[topic][3] in x
                or topic_words[topic][4] in x
            ]

            # If not enough examples, check for second, third, etc. topic loading
            if len(examples_lst) < num_examples:
                extra_examples_lst_2 = [
                    x
                    for x in list(
                        combined_df[combined_df["second_topic"] == topic][
                            text_column_name
                        ].values
                    )
                    if topic_words[topic][0] in x
                    or topic_words[topic][1] in x
                    or topic_words[topic][2] in x
                    or topic_words[topic][3] in x
                    or topic_words[topic][4] in x
                ]
                examples_lst.extend(extra_examples_lst_2)

            # If not enough examples still, check for third topic loading
            if len(examples_lst) < num_examples:
                extra_examples_lst_3 = [
                    x
                    for x in list(
                        combined_df[combined_df["third_topic"] == topic][
                            text_column_name
                        ].values
                    )
                    if topic_words[topic][0] in x
                    or topic_words[topic][1] in x
                    or topic_words[topic][2] in x
                    or topic_words[topic][3] in x
                    or topic_words[topic][4] in x
                ]
                examples_lst.extend(extra_examples_lst_3)

            # Append examples that have one of the top keywords in topic
            sample_texts_lst.append(examples_lst[:num_examples])

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
            topic_words_df["sample_texts"].tolist(),
            index=topic_words_df.index,
        )

        topic_words_explode.columns = [
            "example{}".format(num) for num in range(len(topic_words_explode.columns))
        ]

        all_topics = pd.concat(
            [
                topic_words_df[
                    ["topic_num", "num_in_category", "top_words_and_phrases"]
                ],
                topic_words_explode,
            ],
            axis=1,
        )

        print("Topics created with top words & example texts:")
        print(all_topics)

        original_plus_topics = combined_df[list(X.columns) + ["index", "top_topic_num"]]
        original_with_keywords = pd.merge(
            original_plus_topics,
            all_topics[["topic_num", "top_words_and_phrases"]],
            left_on="top_topic_num",
            right_on="topic_num",
            how="left",
        )[[text_column_name, "topic_num", "top_words_and_phrases"]]

        all_topics.sort_values(by="num_in_category", ascending=False, inplace=True)

        return (
            all_topics,
            original_with_keywords,
            model,
        )

    def coefficients_graph(
        self,
        X_train,
        X_test,
        model,
        model_type,
        filename="shap_graph",
        path="../images/",
        multiclass=False,
        y_test=None,
    ):
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
        multiclass: bool
        y_test: pd.DataFrame
            Only needed for multiclass models
        """
        if not os.path.exists(path):
            os.makedirs(path)

        med = X_train.median().values.reshape((1, X_train.shape[1]))

        # Runs too slow if X_test is huge, take a representative sample
        if X_test.shape[0] > 1000:
            X_test_sample = X_test.sample(1000)
        else:
            X_test_sample = X_test

        if multiclass:
            lst_all_shap_values = []

            for class_num in range(len(y_test.columns)):
                f = lambda x: model.predict_proba(x)[class_num][:, 1]

                explainer = shap.KernelExplainer(f, med)
                shap_values = explainer.shap_values(X_test_sample)
                lst_all_shap_values.append(shap_values)

                class_name = y_test.columns[class_num]
                print(f"SHAP Summary Plot for Class: {class_name}")

                shap.summary_plot(shap_values, X_test_sample)
                plt.tight_layout()
                plt.savefig(f"{path}{class_name}_{filename}.png")

            return np.array(lst_all_shap_values)

        elif model_type.lower() == "classification":
            f = lambda x: model.predict_proba(x)[:, 1]
        else:
            f = lambda x: model.predict(x)

        explainer = shap.KernelExplainer(f, med)
        shap_values = explainer.shap_values(X_test_sample)

        shap.summary_plot(shap_values, X_test_sample)
        plt.tight_layout()
        plt.savefig(f"{path}_{filename}.png")

        return shap_values

    def most_common_words_by_group(
        self,
        X,
        text_col_name,
        group_col_name,
        num_examples,
        num_times_min,
        min_ngram,
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
        words = list(cv.get_feature_names_out())
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
                    ~group_plus_vectors.columns.isin(
                        [
                            text_col_name,
                            "index",
                        ]
                    )
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
        self,
        model_type,
        X_test,
        y_test,
        trained_model,
        csv_file_name="final_model_results",
        multiclass=False,
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
        csv_file_name: str

        Returns
        --------
        model: model
            Fit model
        y_predicted: array
        """
        set_config(print_changed_only=True)

        cleaned_date = (
            datetime.datetime.today().isoformat(" ", "seconds").replace(" ", "-")
        )

        # Predict actual scores
        y_predicted = trained_model.predict(X_test)

        if multiclass:
            pscore = round(precision_score(y_test, y_predicted, average="weighted"), 7)
            rscore = round(recall_score(y_test, y_predicted, average="weighted"), 7)
            ascore = round(accuracy_score(y_test, y_predicted), 7)
            f1score = round(f1_score(y_test, y_predicted, average="weighted"), 7)

            temp_df = pd.DataFrame(
                [
                    [
                        trained_model,
                        pscore,
                        rscore,
                        ascore,
                        f1score,
                    ]
                ],
                columns=[
                    "model",
                    "precision",
                    "recall",
                    "accuracy",
                    "f1",
                ],
            )

            with open(
                f"{'/'.join(csv_file_name.split('/')[:-1])}/model_scoring_crosstabs_{cleaned_date}.txt",
                "w",
            ) as f:
                print(f"Model: {trained_model}", file=f)
                print("\n", file=f)

                print("Overall scores:", file=f)
                print("\n", file=f)
                print(
                    tabulate(
                        temp_df[
                            [
                                "precision",
                                "recall",
                                "accuracy",
                                "f1",
                            ]
                        ],
                        headers="keys",
                        tablefmt="pipe",
                        numalign="left",
                        showindex=False,
                    ),
                    file=f,
                )
                print("\n", file=f)

                # Precision / recall / f1-score for each predicted class
                report_df = (
                    pd.DataFrame(
                        classification_report(
                            y_test,
                            y_predicted,
                            target_names=y_test.columns,
                            output_dict=True,
                        )
                    )
                    .transpose()
                    .drop(["micro avg", "macro avg", "weighted avg", "samples avg"])
                    .drop("support", axis=1)
                )
                print(
                    tabulate(
                        report_df,
                        headers="keys",
                        tablefmt="pipe",
                        numalign="left",
                    ),
                    file=f,
                )
                print("\n", file=f)

                # Counts of predicted vs actuals + true vs false
                confusion_matrix = multilabel_confusion_matrix(y_test, y_predicted)
                matrix_dfs = [
                    pd.DataFrame(
                        matrix,
                        columns=["Predicted False", "Predicted True"],
                        index=["Actual False", "Actual True"],
                    )
                    for matrix in confusion_matrix
                ]

                # Print separately so class name gets printed cleanly first
                for i in range(len(y_test.columns)):
                    print(y_test.columns[i], file=f)
                    print(
                        tabulate(
                            matrix_dfs[i],
                            headers="keys",
                            tablefmt="pipe",
                            numalign="left",
                        ),
                        file=f,
                    )
                    print("\n", file=f)

        elif model_type.lower() == "classification":
            pscore = round(precision_score(y_test, y_predicted), 7)
            rscore = round(recall_score(y_test, y_predicted), 7)
            ascore = round(accuracy_score(y_test, y_predicted), 7)
            f1score = round(f1_score(y_test, y_predicted), 7)

            temp_df = pd.DataFrame(
                [
                    [
                        trained_model,
                        pscore,
                        rscore,
                        ascore,
                        f1score,
                    ]
                ],
                columns=[
                    "model",
                    "precision",
                    "recall",
                    "accuracy",
                    "f1",
                ],
            )

            with open(
                f"{'/'.join(csv_file_name.split('/')[:-1])}/model_scoring_crosstabs_{cleaned_date}.txt",
                "w",
            ) as f:
                print(f"Model: {trained_model}", file=f)
                print("\n", file=f)

                print("Overall scores:", file=f)
                print("\n", file=f)
                print(
                    tabulate(
                        temp_df[
                            [
                                "precision",
                                "recall",
                                "accuracy",
                                "f1",
                            ]
                        ],
                        headers="keys",
                        tablefmt="pipe",
                        numalign="left",
                        showindex=False,
                    ),
                    file=f,
                )
                print("\n", file=f)

                crosstab = pd.crosstab(
                    np.array(y_test),
                    y_predicted,
                )
                class_values = crosstab.columns
                crosstab.columns = [f"Predicted {val}" for val in class_values]
                crosstab.index = [f"Actual {val}" for val in class_values]
                print(
                    tabulate(
                        crosstab,
                        headers="keys",
                        tablefmt="pipe",
                        numalign="left",
                    ),
                    file=f,
                )
                print("\n", file=f)

                sum_crosstab = crosstab.to_numpy().sum()
                prop_crosstab = pd.crosstab(
                    np.array(y_test),
                    y_predicted,
                ).apply(lambda r: round(r / sum_crosstab, 3))
                class_values = prop_crosstab.columns
                prop_crosstab.columns = [f"Predicted {val}" for val in class_values]
                prop_crosstab.index = [f"Actual {val}" for val in class_values]
                print(
                    tabulate(
                        prop_crosstab,
                        headers="keys",
                        tablefmt="pipe",
                        numalign="left",
                    ),
                    file=f,
                )
        else:
            mse = mean_squared_error(y_test, y_predicted)
            mae = median_absolute_error(y_test, y_predicted)
            r2 = round(r2_score(y_test, y_predicted), 7)

            temp_df = pd.DataFrame(
                [
                    [
                        trained_model,
                        round((mse ** 5) * -1, 7),
                        round((mae * -1), 7),
                        r2,
                    ]
                ],
                columns=["model", "negative_rmse", "negative_mae", "r2"],
            )

        temp_df["timestamp"] = cleaned_date

        try:
            previous_df = pd.read_csv(f"{csv_file_name}.csv")

            if temp_df.shape[1] != previous_df.shape[1]:
                print(
                    f"""Unable to save csv because columns do not match. 
                The existing model results csv has these columns: {previous_df.columns}. 
                The new model results csv has these columns: {temp_df.columns}."""
                )
                return trained_model, y_predicted

            final_model_results_df = pd.concat([previous_df, temp_df], axis=0)
            final_model_results_df.reset_index(inplace=True, drop=True)
        except Exception:
            final_model_results_df = temp_df

        final_model_results_df = final_model_results_df.reindex(
            sorted(final_model_results_df.columns), axis=1
        )

        with open(f"{csv_file_name}.csv", "w") as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=",")
            csvwriter.writerow(final_model_results_df.columns)
            for _, row in final_model_results_df.iterrows():
                csvwriter.writerow(row)

        return trained_model, y_predicted

    def coefficients_summary(
        self,
        X,
        y,
        num_repetitions,
        num_coefficients,
        model_type,
        multiclass=False,
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
            coefficients_df.groupby("features").agg(
                [
                    "mean",
                    "median",
                ]
            )
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
        model,
        df,
        X_train,
        X_test,
        id_col,
        num_id_examples,
        num_feature_examples,
        model_type,
        class_names=["False", "True"],
        path="../images/",
    ):
        """
        Uses LIME to inspect an individual prediction and the features that influenced that prediction.

        Parameters
        --------
        model: sklearn model
        df: pd.DataFrame
            Used for getting ids since they aren't typically in training data
        X_train: pd.DataFrame
        X_test: pd.DataFrame
        id_col: str
        num_id_examples: int
        num_feature_examples: int
        model_type: str
            'classification' or 'regression'
        class_names: str
        path: str

        Returns
        --------
        features: list
        """

        if not os.path.exists(path):
            os.makedirs(path)

        def model_preds_adjusted(data):
            if model_type.lower() == "classification":
                predictions = np.array(model.predict_proba(data))
            else:
                predictions = np.array(model.predict(data))
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

        for _ in range(num_id_examples):
            row_idx = random.sample(list(X_test.index), 1)[0]
            exp = explainer.explain_instance(
                np.array(X_test.loc[row_idx]),
                model_preds_adjusted,
                # Include all features
                num_features=len(X_train.columns),
                # Include all classes
                top_labels=len(class_names),
            )

            if model_type.lower() == "classification":
                prediction = class_names[
                    model.predict_proba(pd.DataFrame(X_test.loc[row_idx]).T).argmax()
                ]
            else:
                prediction = round(
                    model.predict(pd.DataFrame(X_test.loc[row_idx]).T)[0], 7
                )

            unique_id = df.loc[row_idx][id_col]
            print(f"\nID: {unique_id}")
            print(f"Prediction: {prediction}\n\n")

            exp_list_all = exp.as_list()

            raw_features = [x[0] for x in exp_list_all]
            raw_values = [x[1] for x in exp_list_all]

            cleaned_features = []

            for feature in exp_list_all:
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
                cleaned_features.append(cleaned_feature_name)

            all_feature_types = X_test.dtypes
            top_feature_types = [
                all_feature_types[feature] for feature in cleaned_features
            ]

            top_features_with_types = [
                [raw_feature, cleaned_feature, feature_type, raw_value]
                for raw_feature, cleaned_feature, feature_type, raw_value in zip(
                    raw_features, cleaned_features, top_feature_types, raw_values
                )
            ]

            i = 0
            for (
                raw_feature,
                cleaned_feature,
                feature_type,
                raw_value,
            ) in top_features_with_types:
                if i > num_feature_examples:
                    break

                actual_feature_val = X_test.loc[row_idx][cleaned_feature]

                # Things that decrease the likelihood of this class are less interesting
                if raw_value < 0:
                    pass
                # Note: uint8 is a bool
                # False bools aren't super interesting
                elif feature_type == "uint8" and actual_feature_val == 0:
                    pass
                # Phrase true bools slightly differently
                elif feature_type == "uint8":
                    print(f"For this id, {cleaned_feature} was true.")

                    print(
                        f"When {cleaned_feature} is true, this increases the likelihood of prediction: {prediction}."
                    )

                    print("\n--------\n")
                    i += 1
                else:
                    print(f"For this id, {cleaned_feature} was {actual_feature_val}.\n")

                    print(
                        f"When {raw_feature}, this increases the likelihood of prediction: {prediction}."
                    )

                    print("\n--------\n")
                    i += 1

            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            fig = plt.figure()
            # Up the image quality to avoid pixelated graphs
            plt.rc("savefig", dpi=300)
            # Limit to top features that can fit cleanly on graph
            exp_list_graph = exp.as_list()[:20]
            vals = [x[1] for x in exp_list_graph]
            # Some labels are really long, shortening them a bit
            names = [x[0][:40] for x in exp_list_graph]
            vals.reverse()
            names.reverse()
            colors = ["green" if x > 0 else "red" for x in vals]
            pos = np.arange(len(exp_list_graph)) + 0.5
            plt.barh(pos, vals, align="center", color=colors)
            plt.yticks(pos, names)
            title = f"id: {unique_id} - Prediction: {prediction}"
            plt.title(title)
            plt.tight_layout()
            # Need bbox to make sure title isn't cut off
            plt.savefig(
                f"../images/lime_graph_id_{unique_id}.png",
                bbox_inches="tight",
                facecolor="white",
            )

        return exp_list_all

    def get_tree_diagram(self, model, X_train, path="../images/"):
        """
        Save a diagram of a trained DecisionTree model

        Parameters
        --------
        model: sklearn model (trained)
        X_train: pd.DataFrame
        path: str
        """
        # Exporting text form of decision tree
        dot_data = export_graphviz(
            model,
            out_file=f"{path}decision-tree.dot",
            feature_names=X_train.columns,
            filled=True,
            rounded=True,
            special_characters=True,
        )
        graph = graphviz.Source(dot_data)

        # Converting text to a visual png file
        os.system(f"dot -Tpng {path}decision-tree.dot -o {path}decision-tree.png")

        # If the file didn't write, try reinstalling graphviz
        if not os.path.exists(f"{path}decision-tree.png"):
            os.system("brew install graphviz")
            os.system(f"dot -Tpng {path}decision-tree.dot -o {path}decision-tree.png")

        return graph
