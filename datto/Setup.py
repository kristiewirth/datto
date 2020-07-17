import logging
import sys
import pandas as pd
import numpy as np

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

    def display_more_data(self, num_to_display):
        """
        Overrides Pandas and Numpy settings to display a larger amount of data instead of only a subset.

        Parameters
        --------
        num_to_display: int
            How many rows/columns to display
        """

        np.set_printoptions(suppress=True, formatter={"float_kind": "{:0.6f}".format})
        pd.set_option("display.float_format", lambda x: "%.6f" % x)

        np.set_printoptions(threshold=num_to_display)
        pd.set_option("display.max_columns", num_to_display)
        pd.set_option("display.max_rows", num_to_display)

