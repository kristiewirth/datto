import logging

from datto.SetupMethods import SetupMethods

s = SetupMethods()


def test_setup_logger():
    logger = s.setup_logger()
    assert isinstance(logger, logging.RootLogger)


def test_display_more_data():
    s.display_more_data(10)
