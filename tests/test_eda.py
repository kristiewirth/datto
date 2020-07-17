import pandas as pd

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


def test_separate_cols_by_type():
    numerical_vals, categorical_vals = eda.separate_cols_by_type(df)

    assert "text" in categorical_vals.columns


def test_check_for_mistyped_booleans():
    boolean_vals = eda.check_for_mistyped_booleans(df)

    assert boolean_vals == ["boolean"]


def test_find_cols_to_exclude():
    cols = eda.find_cols_to_exclude(df)
    assert "group_id" in cols


# TODO: Add a test for sample_unique_vals func


def test_find_correlated_features():
    s = eda.find_correlated_features(df)
    assert s[0] > 0.0
