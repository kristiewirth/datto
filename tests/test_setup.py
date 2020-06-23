import logging

from datto.Setup import Setup

s = Setup()


def test_setup_logger():
    logger = s.setup_logger()
    assert type(logger) == logging.RootLogger


# TODO: Add some text for display all data?
