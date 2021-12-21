import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import progressbar
import seaborn as sns


class Eda:
    """
    Exploratory data analysis (EDA)
    """

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
                and str(col) != "date"
                and "timestamp" not in str(col)
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

    def violin_plots_by_col(self, df, path="../images/", group_by_var=None):
        """
        Makes a violin plot for each numerical column.

        Parameters
        --------
        df: DataFrame
        path: str
        group_by_var: str
            Variable to group violin plots by
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
            # Filter out some extreme outliers for cleaner plot
            filtered_df = df[
                (df[col] <= df[col].quantile(0.99))
                & (df[col] >= df[col].quantile(0.01))
            ]

            fig = plt.figure(figsize=(9, 9))
            ax = fig.add_subplot(111)
            ax.set_title(col)

            if group_by_var:
                sns.violinplot(x=group_by_var, y=col, data=filtered_df, ax=ax)
            else:
                sns.violinplot(
                    x=col,
                    data=filtered_df,
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

            if group_by_var:
                plt.savefig(f"{path}violinplot_{col}_by_{group_by_var}.png")
            else:
                plt.savefig(f"{path}violinplot_{col}.png")

    def bar_graphs_by_col(self, df, path="../images/", group_by_var=None):
        """
        Makes a bar graph for each categorical column.

        Parameters
        --------
        df: DataFrame
        path: str
        group_by_var: str
            Variable to group bar graphs by
        """
        _, categorical_vals = self.separate_cols_by_type(df)

        if categorical_vals.empty:
            return "No categorical columns to graph."

        if not os.path.exists(path):
            os.makedirs(path)

        iter_bar = progressbar.ProgressBar()
        for col in iter_bar(categorical_vals):
            if col == group_by_var:
                continue

            num_unique_vals = len(df[col].unique())

            try:
                if num_unique_vals == 1:
                    continue

                # More values than this doesn't display well, just show the top values
                if group_by_var:
                    # Group bys are hard to read unless this is smaller
                    num_groups = len(df[group_by_var].unique())
                    most_vals_allowed = round(50 / num_groups)
                    if most_vals_allowed < 5:
                        most_vals_allowed = 5
                else:
                    most_vals_allowed = 50

                if num_unique_vals > most_vals_allowed:
                    adjust_vals = df[
                        df[col].isin(
                            [
                                x[0]
                                for x in Counter(df[col]).most_common(most_vals_allowed)
                            ]
                        )
                    ]
                else:
                    adjust_vals = df.copy()

                fig = plt.figure(figsize=(9, 9))
                ax = fig.add_subplot(111)
                ax.set_title(col)

                if group_by_var:
                    # Change to proportions by group instead of straight counts (misleading by sample size)
                    grouped_df = df.groupby([group_by_var, col]).count()
                    grouped_df_pcts = grouped_df.groupby(level=0).apply(
                        lambda x: x / float(x.sum())
                    )

                    grouped_df_pcts = grouped_df_pcts.reset_index()
                    grouped_df_pcts.columns = [group_by_var, col, "proportion"]
                    grouped_df_pcts.sort_values(
                        by="proportion", ascending=True, inplace=True
                    )

                    pivot_df = pd.pivot_table(
                        grouped_df_pcts,
                        values="proportion",
                        index=col,
                        columns=group_by_var,
                    ).reset_index()

                    # Sort pivot table by most common cols
                    sorter = list(
                        adjust_vals.groupby([col])
                        .count()
                        .iloc[:, 1]
                        .sort_values(ascending=True)
                        .index
                    )
                    sorterIndex = dict(zip(sorter, range(len(sorter))))
                    pivot_df["rank"] = pivot_df[col].map(sorterIndex)
                    pivot_df.sort_values(by="rank", ascending=True, inplace=True)
                    pivot_df.drop("rank", axis=1, inplace=True)

                    pivot_df.plot(
                        x=col,
                        kind="barh",
                        ylabel=f"proportion_{col}_within_{group_by_var}",
                        ax=ax,
                    )
                    plt.tight_layout()
                    plt.savefig(
                        f"{path}bargraph_proportion_{col}_within_{group_by_var}.png"
                    )
                else:
                    grouped_df = (
                        adjust_vals.groupby([col]).count().iloc[:, 1]
                        / adjust_vals.shape[0]
                    )
                    grouped_df = grouped_df.reset_index()
                    grouped_df.columns = [col, "proportion"]
                    grouped_df.sort_values(
                        by="proportion", ascending=True, inplace=True
                    )

                    grouped_df.plot(
                        x=col, kind="barh", legend=None, ylabel="proportion", ax=ax
                    )
                    plt.tight_layout()
                    plt.savefig(f"{path}bargraph_{col}.png")
            except Exception:
                continue
