import pandas as pd
import re
import spacy
import numpy as np
import string

nlp = spacy.load("en")


class CleanText:
    def __init__(self):
        pass

    #     # TODO: Fix data import here
    #     # self.all_names = pd.read_pickle("data/all_names")
    #     # def remove_names(self, text):
    #     #     cleaned_text = text
    #     #     for i, row in self.all_names.iterrows():
    #     #         cleaned_text = re.sub(
    #     #             row["name"] + "[^\w\s]*[\s]+", "<NAME> ", cleaned_text
    #     #         )
    #     #     return cleaned_text

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
            "(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})",
            cleaned_text,
        )
        for link in links_found:
            cleaned_text = cleaned_text.replace(link, "<LINK>")
        return cleaned_text

    def lematize(self, text):
        """
        Run `python -m spacy download en` in the terminal first.

        Then, indclue `nlp = spacy.load("en")` in your code.

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
        DataFrame

        Returns
        --------
        DataFrame

        """
        df.rename(columns=lambda x: x.strip().replace(" ", "_").lower(), inplace=True)
        return df

    def remove_duplicate_columns(self, df):
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    def fix_col_data_type(self, df, col, desired_dt):
        if desired_dt in ("float", "int"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif desired_dt == "datetime":
            df[col] = pd.to_datetime(df[col])
        elif desired_dt == "str":
            df[col] = df[col].astype(str)

        return df
