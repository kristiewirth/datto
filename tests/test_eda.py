import pandas as pd
from hypothesis import example, given, strategies
from hypothesis.extra.pandas import column, data_frames

from datto.Eda import Eda

eda = Eda()


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
    assert type(s) == pd.Series
