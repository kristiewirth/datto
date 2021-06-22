import pandas as pd
from hypothesis import given, strategies
from hypothesis.extra.pandas import column, data_frames

from datto.Eda import Eda

eda = Eda()

df = pd.DataFrame(
    [
        ["some text", 1, 1.2],
        ["some other text", 1, 1.4],
        ["i like bananas", 2, 6.5],
        ["i like apples", 2, 7.5],
        ["some text", 1, 1.2],
        ["some other text", 1, 1.4],
        ["i like bananas", 2, 6.5],
        ["i like apples", 2, 7.5],
        ["some text", 1, 1.2],
        ["some other text", 1, 1.4],
        ["i like bananas", 2, 6.5],
        ["i like apples", 2, 7.5],
    ],
    columns=["text", "int", "float"],
)


@given(
    data_frames(
        columns=[column(name="text", dtype=str), column(dtype=int), column(dtype=bool),]
    )
)
def test_separate_cols_by_type(df):
    _, categorical_vals = eda.separate_cols_by_type(df)

    assert "text" in categorical_vals.columns


@given(
    data_frames(
        columns=[
            column(dtype=str),
            column(dtype=int),
            column(
                name="boolean",
                dtype=int,
                elements=strategies.integers(min_value=0, max_value=1),
            ),
        ]
    )
)
def test_check_for_mistyped_booleans(df):
    boolean_vals = eda.check_for_mistyped_booleans(df)

    assert "boolean" in boolean_vals


@given(
    data_frames(
        columns=[
            column(name="group_id", dtype=str),
            column(dtype=int),
            column(dtype=bool),
        ]
    )
)
def test_find_cols_to_exclude(df):
    cols = eda.find_cols_to_exclude(df)
    "group_id" in [[k for k, v in x.items()][0] for x in cols]


@given(
    data_frames(
        columns=[
            column(dtype=str),
            column(dtype=int),
            column(dtype=int),
            column(dtype=bool),
        ]
    )
)
def test_find_correlated_features(df):
    s = eda.find_correlated_features(df)
    assert isinstance(s, pd.DataFrame)


def test_check_unique_by_identifier_col():
    df = pd.DataFrame(["user_1234", "user_32434", "user_1234"], columns=["ids"])
    identifier_col = "ids"
    dup_rows = eda.check_unique_by_identifier_col(df, identifier_col)

    assert not dup_rows.empty


def test_violin_plots_by_col():
    eda.violin_plots_by_col(df)


def test_bar_graphs_by_col():
    eda.bar_graphs_by_col(df)
