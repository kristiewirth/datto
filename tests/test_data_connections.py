import os

from hypothesis import example, given, strategies, settings
from hypothesis.extra.pandas import column, data_frames

from datto.DataConnections import DataConnections, KafkaInterface

dc = DataConnections()


@given(data_frames(columns=[column(dtype=str), column(dtype=int),]))
@settings(deadline=None)
def test_save_to_s3(object_to_save):
    dc.save_to_s3(
        "zapier-data-science-storage/kristie/testing/", object_to_save, "testing-001"
    )


def test_load_from_s3():
    saved_object = dc.load_from_s3(
        "zapier-data-science-storage/kristie/testing/", "testing-001"
    )
    assert saved_object is not None


def test_run_sql_redshift():
    conn = dc.setup_redshift_connection()
    df = dc.run_sql_redshift(conn, """SELECT conv_id FROM hs_convs LIMIT 1""")
    assert ~df.empty


@given(strategies.dictionaries(strategies.text(), strategies.text()))
@settings(deadline=None)
def test_kafka_interface(d):
    ki = KafkaInterface("kristie-testing-1",)
    ki.send([d])
