import numpy as np
import pandas as pd
from datto.CleanDataframe import CleanDataframe
from hypothesis import example, given, strategies
from hypothesis.extra.pandas import column, data_frames

cdf = CleanDataframe()

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


def replace_text_test(df):
    df["text"] = df["text"].apply(lambda x: x.replace("i like", "i love"))
    return df


def test_remove_names():
    text = "Hello John, how are you doing?"
    cleaned_text = cdf.remove_names(text)
    assert "John" not in cleaned_text


@given(strategies.text())
@example("Here's a link: www.google.com Thanks!")
def test_remove_links(text):
    cleaned_text = cdf.remove_links(text)
    assert ".com" not in cleaned_text


@given(strategies.text())
@example("I went running today.")
def test_lematize(text):
    spacy_tokens = cdf.lematize(text)
    assert "ing " not in spacy_tokens


@given(strategies.text())
@example(
    """
    Hello Jane,

    My name is Kristie. I have a question for you.

    Thanks for your help,

    Kristie
    Head of PR at FakeCompany
    """
)
def test_remove_email_greetings_signatures(text):
    cleaned_body = cdf.remove_email_greetings_signatures(text)
    assert "Hello" not in cleaned_body


@given(
    data_frames(
        columns=[
            column(dtype=str),
            column(dtype=int),
            column(dtype=bool),
        ]
    )
)
@example(pd.DataFrame([["Kristie", "Smith"]], columns=["First Name", "Last Name"]))
def test_clean_column_names(df):
    cleaned_df = cdf.clean_column_names(df)
    assert " " not in cleaned_df.columns


@given(
    data_frames(
        columns=[
            column(dtype=str),
            column(dtype=int),
            column(dtype=bool),
        ]
    )
)
@example(
    pd.DataFrame(
        [["Kristie", "Smith", "Smith"]],
        columns=["First Name", "Last Name", "Last Name"],
    )
)
def test_remove_duplicate_columns(df):
    cleaned_df = cdf.remove_duplicate_columns(df)
    assert cleaned_df.shape[1] <= df.shape[1]


# @given(
#     data_frames(
#         columns=[
#             column(name="to_number", dtype=str),
#             column(name="to_datetime", dtype=str),
#             column(name="to_str", dtype=int),
#         ]
#     )
# )
# @example(
#     pd.DataFrame(
#         [["12432", "2012-01-02", 14324]], columns=["to_number", "to_datetime", "to_str"]
#     )
# )
def test_fix_col_data_type():
    simplified_df = pd.DataFrame(
        [["12432", "2012-01-02", 14324]], columns=["to_number", "to_datetime", "to_str"]
    )
    simplified_df = cdf.fix_col_data_type(simplified_df, "to_number", "int")
    simplified_df = cdf.fix_col_data_type(simplified_df, "to_datetime", "datetime")
    simplified_df = cdf.fix_col_data_type(simplified_df, "to_str", "str")

    assert (
        simplified_df.to_number.dtype == "int64"
        or simplified_df.to_number.dtype == "float64"
    )
    assert simplified_df.to_datetime.dtype == "<M8[ns]"
    assert simplified_df.to_str.dtype == "O"


def test_compress_df():
    compressed_df = cdf.compress_df(df)
    assert compressed_df["int"].dtype == "uint8"
    assert compressed_df["float"].dtype == "float32"
    assert compressed_df["text"].dtype == "category"


def test_make_uuid():
    id_num = "609390d88cff44269c2e293bd6b89a0b"
    uuid = cdf.make_uuid(id_num)
    assert uuid == "609390d8-8cff-4426-9c2e-293bd6b89a0b"


def test_batch_pandas_operation():
    num_splits = 2
    identifier_col = "int"
    new_df = cdf.batch_pandas_operation(
        df, num_splits, identifier_col, replace_text_test
    )
    assert not new_df["text"].apply(lambda x: "i like" in x).max()


def test_batch_merge_operation():
    df_1 = df.copy()
    df_2 = df.copy()
    num_splits = 2
    identifier_col = "int"
    merge_col = "int"
    merged_df = cdf.batch_merge_operation(
        df_1, df_2, num_splits, identifier_col, merge_col
    )
    assert merged_df.shape[1] == 5


def test_remove_pii():
    text = """
        Hi my phone number is (123) 456-7890 can you call me
        actually wait it's +1 123 456 7890
        also my email address is cool.guy.123@aol.uk.us.biz.us
        my social is 123-12-1234
        and you can charge this credit card 1234123412341234
        wait actually use my amex 123412345612345
        check out my cool site https://www.stuff.aol.uk.us.biz.us?param=1&other_param=2
        btw, i've submitted like 500 support tickets and i'm furious!
    """
    cleaned_text = cdf.remove_pii(text)
    assert "cool.guy.123" not in cleaned_text
    assert "123" not in cleaned_text
    assert "https" not in cleaned_text
    assert "aol.uk.us.biz.us" not in cleaned_text
    assert "500" in cleaned_text  # we don't want to remove non-PII numbers


def test_fill_nulls_using_regression_model():
    X_train = pd.DataFrame(
        [[1, 4], [2, 4], [4, 3], [1, np.nan]], columns=["col1", "col2"]
    )
    X_test = pd.DataFrame(
        [[1, 2], [2, 4], [1, 3], [4, np.nan]], columns=["col1", "col2"]
    )

    X_train, X_test = cdf.fill_nulls_using_regression_model(X_train, X_test)

    assert not X_train.isnull().values.any()
    assert not X_test.isnull().values.any()
