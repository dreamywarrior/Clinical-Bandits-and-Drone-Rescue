"""Planning algorithms for the drone rescue environment. - PLACEHOLDER
"""

import numpy as np

class ValueIterationPlanner:
    """Planning algorithm placeholder class."""
    # Action mappings used throughout the project.
    ACTIONS = [0,1,2,3,4]
    def __init__():
        # =====================================================
        # VALUE FUNCTION
        # =====================================================
        #
        # Expected Format:
        #
        # np.ndarray:
        #
        # [
        #     [0.0, 0.0, 0.0],
        #     [0.0, 0.0, 0.0],
        # ]
        #
        # Renderer automatically visualizes this.
        #
        # Environment reads this using:
        #     self.value_function
        #
        # =====================================================

        # =====================================================
        # POLICY FORMAT
        # =====================================================
        #
        # Expected Format:
        #
        # {
        #     (0,0): 3,
        #     (0,1): 1,
        #     (0,2): 3,
        # }
        #
        # Meaning:
        #     state -> best action
        #
        # Renderer automatically draws arrows using this.
        #
        # Environment reads this using:
        #     self.policy
        #
        # =====================================================
        return

    def run_value_iteration(self):
        # =====================================================
        # TODO:
        # Implement Value Iteration
        # =====================================================

        return self.value_function, self.policy

    def run_policy_iteration(self):
        # =====================================================
        # TODO:
        # Implement Policy Iteration
        # =====================================================
        return self.value_function, self.policy