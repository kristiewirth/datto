import logging

from datto.SetupMethods import SetupMethods

s = SetupMethods()


def test_setup_logger():
    logger = s.setup_logger()
    assert isinstance(logger, logging.RootLogger)
