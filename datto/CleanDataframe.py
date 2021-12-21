import os
import re
import string

import numpy as np
import pandas as pd
import spacy
from sklearn.linear_model import ElasticNet
from spacy.cli import download

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


class CleanDataframe:
    """
    Clean data using NLP, regex, calculations, etc.
    """

    def remove_names(self, text):
        """
        Parameters
        --------
        text: str

        Returns
        --------
        cleaned_text: str

        """
        all_names = pd.read_pickle(
            os.path.join(os.path.dirname(__file__), "data/all_names")
        )
        cleaned_text = text
        for _, row in all_names.iterrows():
            # Matches name as long as it is not followed by lowercase characters
            # Removing names that are a part of another word
            cleaned_text = re.sub(row["name"] + "(?![a-z])", " ", cleaned_text)
        return cleaned_text

    def remove_pii(self, text):
        """
        Remove common patterns of personally identifiable information (PII)
        Parameters
        --------
        text: str
        Returns
        --------
        cleaned_text: str
        """

        regex_dict = {
            "credit_card_numbers": r"(?:\d[ -]*?){13,16}",
            "phone_numbers": r"[\+]?[\d]{0,3}[\s]?[\(]?\d{3}[\)]?[\s\-\.]{0,1}\d{3}[\s\-\.]{0,1}\d{4}",
            "social_security_numbers": r"(\d{3}[-\s]?\d{2}[-\s]?\d{4})",
            "ip_addresses": r"((?:[0-9]{1,3}\.){3}[0-9]{1,3})",
            "email_addresses": r"([a-zA-Z0-9_\.-]+)@([1-9a-zA-Z\.-]+)\.([a-zA-Z\.]{2,6})",
            "urls": r"((https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,255}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*))",
        }

        cleaned_text = text
        for this_pii_item in regex_dict:
            cleaned_text = re.sub(
                regex_dict[this_pii_item],
                "",
                cleaned_text,
            )

        return cleaned_text

    def remove_links(self, text):
        """
        Parameters
        --------
        text: str

        Returns
        --------
        cleaned_text: str

        """
        cleaned_text = text
        links_found = re.findall(
            r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})",
            cleaned_text,
        )
        for link in links_found:
            cleaned_text = cleaned_text.replace(link, "")
        return cleaned_text

    def lematize(self, text):
        """
        Parameters
        --------
        text: str

        Returns
        --------
        list of spacy tokens

        """
        spacy_text = nlp(text)
        return [token.lemma_ for token in spacy_text if not token.is_space]

    def remove_email_greetings_signatures(self, text):
        """
        In order to obtain the main text of an email only, this method removes greetings, signoffs,
        and signatures by identifying sentences with less than 5% verbs to drop. Does not replace links.

        Inspiration from: https://github.com/mynameisvinn/EmailParser

        Parameters
        --------
        text: str

        Returns
        --------
        text: str

        """
        sentences = text.strip().split("\n")
        non_sentences = []

        for sentence in sentences:
            spacy_text = nlp(sentence.strip())
            verb_count = np.sum(
                [
                    (
                        token.pos_ == "VERB"
                        or token.pos_ == "AUX"
                        or token.pos_ == "ROOT"
                        or token.pos_ == "pcomp"
                    )
                    for token in spacy_text
                ]
            )
            try:
                prob = float(verb_count) / len(spacy_text)
            except Exception:
                prob = 1.0

            # If 5% or less of a sentence is verbs, it's probably not a real sentence
            if prob <= 0.05:
                non_sentences.append(sentence)

        for non_sentence in non_sentences:
            # Don't replace links
            if "http" not in non_sentence and non_sentence not in string.punctuation:
                text = text.replace(non_sentence, "")

        return text

    def clean_column_names(self, df):
        """
        Rename all columns to use underscores to reference columns without bracket formatting

        Parameters
        --------
        df: DataFrame

        Returns
        --------
        df: DataFrame

        """
        df.rename(
            columns=lambda x: str(x).strip().replace(" ", "_").lower(), inplace=True
        )
        return df

    def remove_duplicate_columns(self, df):
        """
        Remove columns with the same name

        Parameters
        --------
        df: DataFrame

        Returns
        --------
        df: DataFrame

        """
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    def fix_col_data_type(self, df, col, desired_dt):
        """
        Change column datatype using the best method for each type.

        Parameters
        --------
        df: DataFrame
        col: str
            Column to change the dtype for
        desired_dt: str
            {'float', 'int', 'datetime', 'str'}

        Returns
        --------
        df: DataFrame

        """
        if desired_dt in ("float", "int"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif desired_dt == "datetime":
            df[col] = pd.to_datetime(df[col], errors="coerce")
        elif desired_dt == "str":
            df[col] = df[col].astype(str)

        return df

    def compress_df(self, df):
        """
        Compresses each dataframe column as much as possible depending on type and values.

        Parameters
        --------
        df: DataFrame

        Returns
        --------
        df: DataFrame

        """
        for col in df.columns:
            if df[col].dtype == "O":
                unique_vals = df[col].nunique()
                count_vals = df[col].shape[0]
                if unique_vals < (count_vals * 0.5):
                    df[col] = df[col].astype("category")
            elif df[col].dtype == "int64":
                df[col] = pd.to_numeric(df[col], downcast="unsigned")
            elif df[col].dtype == "float64":
                df[col] = pd.to_numeric(df[col], downcast="float")

        return df

    def make_uuid(self, id_num):
        """
        Make a UUid_num from a text string

        Parameters
        --------
        id_num: str

        Returns
        --------
        uuid: str

        """
        return (
            id_num[:8]
            + "-"
            + id_num[8:12]
            + "-"
            + id_num[12:16]
            + "-"
            + id_num[16:20]
            + "-"
            + id_num[20:]
        )

    def batch_pandas_operation(self, df, num_splits, identifier_col, func):
        """
        Use a function on a Pandas DataFrame in chunks for faster processing.

        Parameters
        --------
        df: DataFrame
        num_splits: int
        identifier_col: str
        func: Function

        Returns
        --------
        new_df: DataFrame

        """
        new_df = pd.DataFrame()
        ids = df[identifier_col].unique()
        division = len(ids) / float(num_splits)
        part_ids = [
            ids[int(round(division * i)) : int(round(division * (i + 1)))]
            for i in range(num_splits)
        ]
        for lst in part_ids:
            temp_df = df[df[identifier_col].isin(lst)]
            new_df = new_df.append(func(temp_df))

        return new_df

    def batch_merge_operation(self, df_1, df_2, num_splits, identifier_col, merge_col):
        """
        Merge two Pandas DataFrame in chunks for faster processing.

        Parameters
        --------
        df_1: DataFrame
        df_2: DataFrame
        num_splits: int
        identifier_col: str
        merge_col: str

        Returns
        --------
        new_df: DataFrame

        """
        new_df = pd.DataFrame()
        ids = df_1[identifier_col].unique()
        division = len(ids) / float(num_splits)
        part_ids = [
            ids[int(round(division * i)) : int(round(division * (i + 1)))]
            for i in range(num_splits)
        ]
        for lst in part_ids:
            temp_df = df_1[df_1[identifier_col].isin(lst)]
            temp_df = pd.merge(temp_df, df_2, how="left", on=merge_col)
            new_df = new_df.append(temp_df)
        return new_df

    def fill_nulls_using_regression_model(self, X_train, X_test):
        """
        Trains a regression model on non-null data and predicts values to fill in nulls

        Parameters
        --------
        X_train: pd.DataFrame
        X_test: pd.DataFrme


        Returns
        --------
        X_train: pd.DataFrame
        X_test: pd.DataFrame

        """
        numerical_vals = X_train.select_dtypes(exclude=["object", "bool"])

        # Filling null values using a smaller regression model
        for column in numerical_vals.columns:
            # Getting indices in which the given column is not null
            filled = list(X_train[column].dropna().index)

            # Getting all the X train data for the rows in which the given column is not blank
            mini_training_data = (
                X_train.drop(column, axis=1)
                .apply(lambda x: x.fillna(x.mean()), axis=0)
                .iloc[filled]
            )
            # Getting the column to be filled data where it is not null
            mini_training_target = X_train[column].iloc[filled]

            # Instantiating model to predict values
            model = ElasticNet(alpha=0.1, l1_ratio=0.5)
            model.fit(mini_training_data, mini_training_target)

            # Getting indices in which the given column is null
            nulls = [x for x in X_train.index if x not in filled]

            # Getting all the X train data for the rows in which the given column has blank values
            mini_testing_data1 = (
                X_train.drop(column, axis=1)
                .apply(lambda x: x.fillna(x.mean()), axis=0)
                .iloc[nulls]
            )
            if mini_testing_data1.empty == False:
                # Predicting the values of the given column where it is blank
                predictions1 = model.predict(mini_testing_data1)
                # Filling in the values that are blank using the predictions
                X_train[column].iloc[nulls] = predictions1

            # Repeating the process for X test (but just using the already trained model)
            nulls = [x for x in X_test.index if x not in filled]

            # Getting all the X test data for the rows in which the given column has blank values
            mini_testing_data2 = (
                X_test.drop(column, axis=1)
                .apply(lambda x: x.fillna(x.mean()), axis=0)
                .iloc[nulls]
            )

            if mini_testing_data2.empty == False:
                # Predicting the values of the given column where it is blank
                predictions2 = model.predict(mini_testing_data2)
                # Filling in the values that are blank using the predictions
                X_test[column].iloc[nulls] = predictions2

        return X_train, X_test
