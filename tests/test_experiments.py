import numpy as np

from datto.Experiments import Experiments

e = Experiments()


def test_assign_condition_by_id():
    user_id = 12
    conditions = np.array(["exclude", "treatment", "control"])
    proportions_by_conditions = np.array([0.3, 0.3, 0.4])
    random_state = 199
    chosen_condition_1 = e.assign_condition_by_id(
        user_id, conditions, proportions_by_conditions, random_state
    )
    chosen_condition_2 = e.assign_condition_by_id(
        user_id, conditions, proportions_by_conditions, random_state
    )
    chosen_condition_3 = e.assign_condition_by_id(
        user_id, conditions, proportions_by_conditions, random_state
    )

    assert chosen_condition_1 == chosen_condition_2 == chosen_condition_3
