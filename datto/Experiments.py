import numpy as np
import hashlib


class Experiments:
    def __init__(self):
        pass

    def assign_condition_by_id(
        self, id, conditions, proportions_by_conditions, random_state
    ):
        """
        Assign a given id to the same experimental condition every time for a consistent user experience.
        I.e. customer #15 will always be in the treatment condition.

        Parameters
        --------
        id: int
        conditions: numpy array
            E.g. ['exclude', 'treatment', 'control']
        proportions_by_conditions: numpy array
            Should add up to 1, e.g. [0.8, 0.1, 0.1]
        random_state: int
            Divisor used for consistent assignment

        Returns
        --------
        chosen_condition: str
            Chooses one of the conditions you provided

        """
        assert len(conditions) == len(
            proportions_by_conditions
        ), "Need a proportion of assignment for each condition (and vice versa)."

        assert (
            proportions_by_conditions.sum() == 1.0
        ), "Need proportions to add up to 1."

        md5_result = hashlib.md5(str(id).encode())
        hex_string = md5_result.hexdigest()

        # Each hexadecimal character carries 4 bits of information.
        # The integers in Python are 32 bits or 64 bits depending on system architecture.
        # To be safe, we'll assume a 32 bit architecure, even though it is almost certainly 64 bits.
        # That means we can process only 8 characters of hex into int without fear of losing fidelity.
        hex_string_truncated = hex_string[-8:]

        # Hexadecimal is a base 16 representation, so convert the hex characters to integers.
        numeric_result = int(hex_string_truncated, 16)

        # From the numeric, select a condition.
        # We force this integer between 0 and random_state-1 via the modulo.
        remainder = numeric_result % random_state
        thresholds_for_condition_assignment = np.floor(
            proportions_by_conditions.cumsum() * random_state
        )

        # Get the first index where the remainder is less than the condition boundry
        condition_index = np.where(remainder < thresholds_for_condition_assignment)[0][
            0
        ]
        chosen_condition = conditions[condition_index]

        return chosen_condition
