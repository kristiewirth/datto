from datto.DataConnections import DataConnections

dc = DataConnections()


def test_save_to_s3():
    dc.save_to_s3(
        "zapier-data-science-storage/kristie/testing/", [1, 2, 3], "testing-001"
    )


def test_load_from_s3():
    lst = dc.load_from_s3("zapier-data-science-storage/kristie/testing/", "testing-001")
    assert lst == [1, 2, 3]


# TODO: Write tests for sql functions that don't include any credentials in this repo
