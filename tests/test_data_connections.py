import os

from hypothesis import example, given, strategies, settings
from hypothesis.extra.pandas import column, data_frames

from datto.DataConnections import S3Connections, SQLConnections

s3 = S3Connections()


@given(data_frames(columns=[column(dtype=str), column(dtype=int),]))
@settings(deadline=None)
def test_save_to_s3(object_to_save):
    s3.save_to_s3(
        "zapier-data-science-storage/kristie/testing/", object_to_save, "testing-001"
    )


def test_load_from_s3():
    saved_object = s3.load_from_s3(
        "zapier-data-science-storage/kristie/testing/", "testing-001"
    )
    assert saved_object is not None


def test_run_sql_redshift():
    sql = SQLConnections()
    df = sql.run_sql_redshift("""SELECT conv_id FROM hs_convs LIMIT 1""")
    assert ~df.empty
