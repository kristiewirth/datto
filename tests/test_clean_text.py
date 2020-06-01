from datto.CleanText import CleanText
import spacy
import pandas as pd

ct = CleanText()

# def test_remove_names():
#     text = "Hello John, how are you doing?"
#     cleaned_text = ct.remove_names(text)
#     assert "John" not in cleaned_text


def test_remove_links():
    text = "Here's a link: www.google.com Thanks!"
    cleaned_text = ct.remove_links(text)
    assert "www.google.com" not in cleaned_text


def test_lematize():
    text = "I went running today."
    spacy_tokens = ct.lematize(text)
    assert "running" not in spacy_tokens


def test_remove_email_greetings_signatures():
    text = """
    Hello Jane,

    My name is Kristie. I have a question for you.

    Thanks for your help,

    Kristie
    Head of PR at FakeCompany
    """
    cleaned_body = ct.remove_email_greetings_signatures(text)
    assert "Head of PR" not in cleaned_body


def test_clean_column_names():
    df = pd.DataFrame([["Kristie", "Smith"]], columns=["First Name", "Last Name"])
    cleaned_df = ct.clean_column_names(df)
    assert "First Name" not in cleaned_df.columns


def test_remove_duplicate_columns():
    df = pd.DataFrame(
        [["Kristie", "Smith", "Smith"]],
        columns=["First Name", "Last Name", "Last Name"],
    )
    cleaned_df = ct.remove_duplicate_columns(df)
    assert cleaned_df.shape[1] == 2


def test_fix_col_data_type():
    df = pd.DataFrame(
        [["12432", "2012-01-02", 14324]], columns=["to_number", "to_datetime", "to_str"]
    )
    df = ct.fix_col_data_type(df, "to_number", "int")
    df = ct.fix_col_data_type(df, "to_datetime", "datetime")
    df = ct.fix_col_data_type(df, "to_str", "str")

    assert df.to_number.dtype == "int64"
    assert df.to_datetime.dtype == "<M8[ns]"
    assert df.to_str.dtype == "O"
