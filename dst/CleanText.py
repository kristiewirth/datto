import pandas as pd
import re


class CleanText:
    def __init__(self):
        self.all_names = pd.read_pickle("data/all_names")

    def removeNames(self, text):
        cleaned_text = text
        for i, row in self.all_names.iterrows():
            cleaned_text = re.sub(
                row["name"] + "[^\w\s]*[\s]+", "<NAME> ", cleaned_text
            )
        return cleaned_text

    def removeLinks(self, text):
        cleaned_text = text
        links_found = re.findall(
            "(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})",
            cleaned_text,
        )

        for link in links_found:
            cleaned_text = cleaned_text.replace(link, "<LINK>")

        return cleaned_text
