import logging
import sys

from pythonjsonlogger import jsonlogger


class Setup:
    def setup_logger(self):
        """
        Returns
        --------
        logger: jsonlogger
        """
        logger = logging.getLogger()
        logging.basicConfig(
            format="%(levelname) -10s %(asctime)s %(module)s at line %(lineno)d: %(message)s",
            level=logging.DEBUG,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logHandler = logging.StreamHandler(sys.stdout)
        formatter = jsonlogger.JsonFormatter()
        logHandler.setFormatter(formatter)
        logger.addHandler(logHandler)
        return logger
