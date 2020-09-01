import string
from math import ceil
from operator import itemgetter

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, nmf
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from datto.CleanText import CleanText


class ModelResults:
    def most_similar_texts(self, X, num_examples, text_column_name, num_topics=None):
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
        )

        ct = CleanText()
        vectorizer = TfidfVectorizer(
            tokenizer=ct.lematize, ngram_range=(1, 3), stop_words=all_stop_words,
        )
        vectors = vectorizer.fit_transform(X[text_column_name]).todense()

        # Adding words/phrases used in text data frequencies back into the dataset (so we can see feature importances later)
        vocab = vectorizer.get_feature_names()
        vector_df = pd.DataFrame(vectors, columns=vocab, index=X.index)

        if not num_topics:
            if X.shape[0] < 20:
                return "Too few examples to categorize."

            # In case 1, add 1 to get at least 2
            # The rest are based on eyeballing numbers
            min_topics = ceil(X.shape[0] * 0.01) + 1
            max_topics = ceil(X.shape[0] * 0.2)
            step = ceil((max_topics - min_topics) / 5)

            topic_nums = list(np.arange(min_topics, max_topics, step))

            proportions_list = []

            texts = X[text_column_name].apply(lambda x: ct.lematize(x))

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
                    model=model, texts=texts, dictionary=dictionary, coherence="c_v"
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

        topic_words_df["topic_num"] = [x for x in topic_words.keys()]
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

        return (
            concated_topics,
            combined_df[["index", text_column_name, "top_topic_num"]],
        )

    def coefficients_summary(
        self, X, y, num_repetitions, num_coefficients, model_type, params={}
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
        params: dict
            Optional - add to change regression params, otherwise use default

        Returns
        --------
        simplified_df: DataFrame
            Has mean, median, and standard deviation for coefficients after several runs

        """
        coefficients_df = pd.DataFrame(columns=["features", "coefficients"])

        if model_type == "classification":
            model = LogisticRegression(**params)
        else:
            model = ElasticNet(**params)

        for i in range(num_repetitions):
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            model.fit(X_train, y_train)
            coefs = model.coef_[0]

            temp_df = pd.DataFrame(
                [x for x in zip(X_train.columns, coefs)],
                columns=["features", "coefficients"],
            )

            coefficients_df = coefficients_df.append(temp_df)

        column_of_interest = "coefficients"

        summary_coefficients_df = pd.DataFrame(
            coefficients_df.groupby("features").agg(
                {column_of_interest: ["mean", "std", "median"]}
            )
        )

        summary_coefficients_df.columns = ["mean", "std", "median"]

        summary_coefficients_df.reset_index(inplace=True)

        value_counts_df = pd.DataFrame(columns=["features"])
        for col in X_train.columns:
            temp_df = pd.DataFrame([[col,]], columns=["features"],)
            value_counts_df = value_counts_df.append(temp_df)
        value_counts_df.reset_index(inplace=True, drop=True)

        combined_df = summary_coefficients_df.merge(
            value_counts_df, on="features", how="left"
        )

        combined_df["abs_val_mean"] = abs(combined_df["mean"])

        combined_df.sort_values("abs_val_mean", inplace=True, ascending=False)

        simplified_df = (
            combined_df[["features", "mean", "std", "median"]]
            .head(num_coefficients)
            .round(7)
        )

        # 7 is the most before it flips back to scientific notation
        print("Coefficients summary (descending by mean abs value):")
        print(simplified_df)

        return simplified_df

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
        min_n_gram: int

        Returns
        --------
        overall_counts_df: DataFrame
            Has groups, top words, and counts

        """
        all_stop_words = (
            set(ENGLISH_STOP_WORDS)
            | set(["-PRON-"])
            | set(string.punctuation)
            | set([" "])
        )

        cv = CountVectorizer(
            stop_words=all_stop_words, ngram_range=(min_ngram, 3), min_df=num_times_min,
        )
        vectors = cv.fit_transform(X[text_col_name]).todense()
        words = cv.get_feature_names()
        vectors_df = pd.DataFrame(vectors, columns=words)

        group_plus_vectors = pd.concat([vectors_df, X.reset_index(drop=False)], axis=1)

        count_words = pd.DataFrame(
            group_plus_vectors.groupby(group_col_name).count()["index"]
        )
        count_words.columns = ["count"]

        group_plus_vectors = group_plus_vectors.merge(
            count_words, on=group_col_name, how="left"
        )

        group_plus_vectors["count"].fillna(0, inplace=True)

        sums_by_col = (
            group_plus_vectors[
                group_plus_vectors.columns[
                    ~group_plus_vectors.columns.isin([text_col_name, "index",])
                ]
            ]
            .groupby(group_col_name)
            .sum()
        )

        sums_by_col.sort_values(by="count", ascending=False, inplace=True)

        sums_by_col.drop("count", axis=1, inplace=True)

        array_sums = np.array(sums_by_col)
        sums_values_descending = -np.sort(-array_sums, axis=1)
        sums_indices_descending = (-array_sums).argsort()

        highest_sum = pd.DataFrame(sums_values_descending[:, 0])
        highest_sum.columns = ["highest_sum"]

        sums_by_col["highest_sum"] = highest_sum["highest_sum"].values

        overall_counts_df = pd.DataFrame(columns=["group_name", "top_words_and_counts"])
        i = 0
        for row in sums_by_col.index:
            dict = {}
            temp_df = pd.DataFrame(columns=["group_name", "top_words_and_counts"])
            temp_df["group_name"] = [row]
            top_columns = sums_by_col.columns[
                sums_indices_descending[i][:num_examples]
            ].values
            top_counts = sums_values_descending[i][:num_examples]
            [dict.update({x: y}) for x, y in zip(top_columns, top_counts)]
            temp_df["top_words_and_counts"] = [dict]
            overall_counts_df = overall_counts_df.append([temp_df])
            print(f"Group Name: {row}\n")
            for k, v in dict.items():
                print(k, v)
            print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
            i += 1

        return overall_counts_df

    def score_final_model(
        self, model_type, X_train, y_train, X_test, y_test, full_pipeline
    ):
        """
        Score your model on the test dataset. Only run this once to get an idea of how your model will perform in realtime.
        Run it after you have chosen your model & parameters to avoid problems with overfitting.

        Parameters
        --------
        model_type: str
        X_train: DataFrame
        y_train: DataFrame
        X_test: DataFrame
        y_test: DataFrame
        full_pipeline: sklearn Pipeline

        Returns
        --------
        full_pipeline: model
            Fit model
        y_predicted: array
        """
        # Fitting the final model
        full_pipeline.fit(X_train, y_train)

        # Predict actual scores
        y_predicted = full_pipeline.predict(X_test)

        if model_type == "classification":
            pscore = precision_score(y_test, y_predicted)
            rscore = recall_score(y_test, y_predicted)
            roc_auc = roc_auc_score(y_test, y_predicted)

            print(f"Final Model Precision: {pscore}")
            print(f"Final Model Recall: {rscore}")
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
            r2 = r2_score(y_test, y_predicted)

            print(f"Mean Negative Root Mean Squared Errror: {mse}")
            print(f"Mean Negative Median Absolute Error: {mae}")
            print(f"Mean R2: {r2}")

        return full_pipeline, y_predicted
