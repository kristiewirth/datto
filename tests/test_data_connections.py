from datto.DataConnections import DataConnections
from datto.DataConnections import KafkaInterface
import os

dc = DataConnections()


def test_save_to_s3():
    dc.save_to_s3(
        "zapier-data-science-storage/kristie/testing/", [1, 2, 3], "testing-001"
    )


def test_load_from_s3():
    lst = dc.load_from_s3("zapier-data-science-storage/kristie/testing/", "testing-001")
    assert lst == [1, 2, 3]


def test_run_sql_redshift():
    conn = dc.setup_redshift_connection()
    df = dc.run_sql_redshift(conn, """SELECT conv_id FROM hs_convs LIMIT 1""")
    assert ~df.empty


def test_kafka_interface():
    ki = KafkaInterface("kristie-testing-1",)
    ki.send([{"testa": "a", "testb": "b"}])
