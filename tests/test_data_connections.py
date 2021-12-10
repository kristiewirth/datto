import os

import pandas as pd
from datto.DataConnections import (
    NotebookConnections,
    S3Connections,
    SlackConnections,
    SQLConnections,
)
from hypothesis import given, settings
from hypothesis.extra.pandas import column, data_frames

s3 = S3Connections()
is_travis = "TRAVIS" in os.environ


@given(
    data_frames(
        columns=[
            column(dtype=str),
            column(dtype=int),
        ]
    )
)
@settings(deadline=None)
def test_save_to_s3(object_to_save):
    if not is_travis:
        s3.save_to_s3(
            "zapier-data-science-storage/kristie/testing/",
            object_to_save,
            "testing-001",
        )


def test_load_from_s3():
    if not is_travis:
        saved_object = s3.load_from_s3(
            "zapier-data-science-storage/kristie/testing/", "testing-001"
        )
        assert saved_object is not None


def test_run_sql_redshift():
    if not is_travis:
        sql = SQLConnections()
        df = sql.run_sql_redshift("""SELECT conv_id FROM hs_convs LIMIT 1""")
        assert ~df.empty


nc = NotebookConnections()


def test_save_as_script():
    # Remove previous file if it exists to properly test function
    try:
        os.system(
            "rm /Users/kristiewirth/Documents/work/datto/files_to_ignore/start_notebook.py"
        )
    except Exception:
        pass

    nc.save_as_script(
        "/Users/kristiewirth/Documents/work/datto/files_to_ignore/start_notebook.ipynb"
    )

    saved_script = open(
        "/Users/kristiewirth/Documents/work/datto/files_to_ignore/start_notebook.py"
    ).read()

    assert saved_script is not None


def test_save_as_notebook():
    # Remove previous file if it exists to properly test function
    try:
        os.system(
            "rm /Users/kristiewirth/Documents/work/datto/files_to_ignore/start_script.ipynb"
        )
    except Exception:
        pass

    nc.save_as_notebook(
        "/Users/kristiewirth/Documents/work/datto/files_to_ignore/start_script.py"
    )

    saved_notebook = open(
        "/Users/kristiewirth/Documents/work/datto/files_to_ignore/start_script.ipynb"
    ).read()

    assert saved_notebook is not None


slack_dc = SlackConnections()


def test_get_messages():
    messages_df = slack_dc.get_messages(
        ["C028W5V2A"],
        messages_limit=1000,
    )

    assert ~messages_df.empty


def test_get_messages_wrong_channel():
    messages_df = slack_dc.get_messages(
        ["12342342"],
        messages_limit=1000,
    )

    assert messages_df is not pd.DataFrame


def test_get_messages_wrong_token():
    slack_wrong = SlackConnections(slack_api_token="not_valid_token")

    messages_df = slack_wrong.get_messages(
        ["C028W5V2A"],
        messages_limit=10,
    )

    assert messages_df is not pd.DataFrame
