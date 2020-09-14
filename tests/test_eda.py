import pandas as pd
from hypothesis import example, given, strategies
from hypothesis.extra.pandas import column, data_frames

from datto.Eda import Eda

eda = Eda()

df = pd.DataFrame(
    [
        ["some text", 1, 0],
        ["some other text", 1, 0],
        ["i like bananas", 2, 1],
        ["i like apples", 2, 0],
        ["running is fun", 3, 0],
        ["jumping is fun", 3, 1],
        ["do you read mysteries?", 4, 0],
        ["do you read nonfiction?", 4, 1],
    ],
    columns=["text", "group_id", "boolean"],
)


@given(
    data_frames(
        columns=[column(name="text", dtype=str), column(dtype=int), column(dtype=bool),]
    )
)
def test_separate_cols_by_type(df):
    numerical_vals, categorical_vals = eda.separate_cols_by_type(df)

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
    assert "group_id" in cols


# TODO: Add a test for sample_unique_vals func


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
    assert type(s) == pd.Series
