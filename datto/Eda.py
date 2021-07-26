import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import progressbar
import seaborn as sns


class Eda:
    def separate_cols_by_type(self, df):
        """
        Split the DataFrame into two groups by type

        Parameters
        --------
        df: DataFrame

        Returns
        --------
        numerical_vals: DataFrame
        categorical_vals: DataFrame
        """
        numerical_vals = df[
            [
                col
                for col in df.select_dtypes(
                    exclude=["object", "bool", "datetime"]
                ).columns
                # ID columns values aren't particularly important to examine
                if "_id" not in str(col)
            ]
        ]

        categorical_vals = df[
            [
                col
                for col in df.select_dtypes(include=["object", "bool"]).columns
                if "_id" not in str(col)
            ]
        ]

        return numerical_vals, categorical_vals

    def check_for_mistyped_cols(self, numerical_vals, categorical_vals):
        """
        Check for columns coded incorrectly

        Parameters
        --------
        numerical_vals: list
        categorical_vals: list

        Returns
        --------
        mistyped_cols: list

        """
        mistyped_cols = []

        for col in numerical_vals.columns:
            if numerical_vals[col].nunique() <= 20:
                print("Coded as numerical, is this actually an object / bool?\n")
                print(col)
                print(numerical_vals[col].unique())
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                mistyped_cols.append(col)

        for col in categorical_vals.columns:
            if "_id" in col:
                continue
            # Booleans can be recoded as floats but still are good as booleans
            elif categorical_vals[col].dtypes == bool:
                continue
            try:
                # Test two random values
                float(categorical_vals[col][0])
                float(categorical_vals[col][5])
                print("Coded as categorical, is this actually an int / float?\n")
                print(col)
                print(categorical_vals[col].unique()[:10])
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                mistyped_cols.append(col)
            except Exception:
                pass

        return mistyped_cols

    def find_cols_to_exclude(self, df):
        """
        Returns columns that may not be helpful for model building.

        Exclusion criteria:
        - Possible PII (address, name, username, date, etc. in col name)
        - Large proportion of nulls
        - Only 1 value in entire col
        - Dates
        - Low variance in col values
        - Large number of categorical values

        Parameters
        --------
        df: DataFrame

        Returns
        --------
        lst: list

        """
        lst = []
        for col in df.columns:
            if (
                "address" in str(col)
                or "first_name" in str(col)
                or "last_name" in str(col)
                or "username" in str(col)
                or "_id" in str(col)
                or "date" in str(col)
                or "time" in str(col)
            ):
                lst.append({col: "Considering excluding because potential PII column."})
            elif df[col].isnull().sum() / float(df.shape[0]) >= 0.5:
                lst.append(
                    {
                        col: "Considering excluding because {}% of column is null.".format(
                            round(
                                (df[col].isnull().sum() / float(df.shape[0]) * 100.0), 2
                            )
                        )
                    }
                )
            elif len(df[col].unique()) <= 1:
                lst.append(
                    {
                        col: "Considering excluding because column includes only one value."
                    }
                )
            elif df[col].dtype == "datetime64[ns]":
                lst.append(
                    {col: "Considering excluding because column is a timestamp."}
                )
            elif df[col].dtype not in ["object", "bool"]:
                if df[col].var() < 0.00001:
                    lst.append(
                        {
                            col: "Considering excluding because column variance is low ({})".format(
                                round(df[col].var(), 2)
                            )
                        }
                    )
            elif df[col].dtype in ["object", "bool"]:
                if len(df[col].unique()) > 500:
                    lst.append(
                        {
                            col: "Considering excluding because object column has large number of unique values ({})".format(
                                len(df[col].unique())
                            )
                        }
                    )

        [print(x) for x in lst]

        return lst

    def sample_unique_vals(self, df):
        """
        Examine a few unique vals in each column

        Parameters
        --------
        df: DataFrame
        """
        for col in df:
            print(col)
            try:
                print(df[col].unique()[:20])
                print(df[col].nunique())
            except Exception:
                pass
            print("\n------------------------------------\n")

    def find_correlated_features(self, df):
        """
        Find & sort correlated features

        Parameters
        --------
        df: DataFrame

        Returns
        --------
        s: Series

        """
        if df.empty:
            return pd.DataFrame()

        c = df.corr().abs()
        s = c.unstack()
        s = s[s <= 0.99999]
        s = s.sort_values(ascending=False)
        s_df = s.reset_index()
        s_df.columns = ["feature_1", "feature_2", "corr"]
        return s_df

    def check_unique_by_identifier_col(self, df, identifier_col):
        """
        Check if there are duplicates by entity (e.g. user, item).

        Parameters
        --------
        df: DataFrame

        Returns
        --------
        dup_rows: DataFrame

        """

        try:
            dup_rows = pd.concat(
                x for col, x in df.groupby(identifier_col) if len(x) > 1
            ).sort_values(identifier_col)
        except Exception:
            return "No duplicate rows found."

        return dup_rows

    def violin_plots_by_col(self, df, path="../images/"):
        """
        Makes a violin plot for each numerical column.

        Parameters
        --------
        df: DataFrame
        """

        numerical_vals, _ = self.separate_cols_by_type(df)

        # Need to fill zeros to get accurate percentile numbers
        numerical_vals.fillna(0, inplace=True)

        if numerical_vals.empty:
            return "No numerical columns to graph."

        if not os.path.exists(path):
            os.makedirs(path)

        iter_bar = progressbar.ProgressBar()
        for col in iter_bar(numerical_vals):
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111)
            ax.set_title(col)

            sns.violinplot(
                x=numerical_vals[col],
                ax=ax,
            )

            text = "75th Percentile: {}\nMedian: {}\n25th Percentile: {}".format(
                round(np.percentile(numerical_vals[col], 75), 2),
                round(np.median(numerical_vals[col]), 2),
                round(np.percentile(numerical_vals[col], 25), 2),
            )

            # Place a text box in upper left in axes coords
            props = dict(boxstyle="round", facecolor="white", alpha=0.5)
            ax.text(
                0.05,
                0.95,
                text,
                transform=ax.transAxes,
                fontsize=14,
                verticalalignment="top",
                bbox=props,
            )
            plt.tight_layout()

            plt.savefig(f"{path}violinplot_{col}.png")

    def bar_graphs_by_col(self, df, path="../images/"):
        """
        Makes a bar graph for each categorical column.

        Parameters
        --------
        df: DataFrame
        """
        _, categorical_vals = self.separate_cols_by_type(df)

        if categorical_vals.empty:
            return "No categorical columns to graph."

        if not os.path.exists(path):
            os.makedirs(path)

        iter_bar = progressbar.ProgressBar()
        for col in iter_bar(categorical_vals):
            try:
                if len(df[col].unique()) == 1:
                    continue

                # More values than this doesn't display well, just show the top values
                if len(df[col].unique()) > 45:
                    adjust_vals = df[
                        df[col].isin([x[0] for x in Counter(df[col]).most_common(45)])
                    ][col]
                else:
                    adjust_vals = df[col]

                fig = plt.figure(figsize=(7, 7))
                ax = fig.add_subplot(111)
                ax.set_title(col)

                counts_df = adjust_vals.value_counts().reset_index()
                counts_df.sort_values(by=col, ascending=False, inplace=True)

                if adjust_vals.dtypes == bool:
                    # For some reason seaborn flips these incorrectly for boolean variables
                    sns.barplot(
                        x=counts_df["index"],
                        y=counts_df[col],
                        ax=ax,
                    )

                else:
                    sns.barplot(
                        x=counts_df[col],
                        y=counts_df["index"],
                        ax=ax,
                    )

                ax.set_xlabel("Count")
                ax.set_ylabel("Values")

                plt.tight_layout()
                plt.savefig(f"{path}bargraph_{col}.png")
            except Exception:
                continue
