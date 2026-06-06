"""
=====================================================================
Expected Outcome 1: Custom Drone Rescue Environment
=====================================================================
Gymnasium-compatible environment with reset(), step(), and render().
Handles battery updates, rescue-target removal, stochastic wind
movement, charging stations, blocked cells, and reward computation.

This file is separated from the main DP file to keep concerns clean:
  - gymnasium_env.py  →  environment definition (this file)
  - TEAM_181 - DP.py  →  DP algorithms + analysis (main execution)

To run the DP analysis, only the main file is needed.
This file is required only when using the Gymnasium environment
directly (e.g. for interactive simulation via Drone_Rescue_DP/).
=====================================================================
"""

import os
import sys
import numpy as np

from gymnasium import Env
from gymnasium.spaces import Box, Discrete

# Shared constants (same values as the main DP file)
GRID_MAP = [
    [1,  0,  2,  0,  5],
    [0, -1,  3,  0,  2],
    [0,  0,  4,  0,  0],
    [3,  0, -1,  0,  0],
    [5,  0,  0,  3,  0],
]

SAFE       =  0
START      =  1
WIND       =  2
DANGER     =  3
CHARGING   =  4
RESCUE     =  5
BLOCKED    = -1

MAX_BATTERY  = 15
BATTERY_COST = 1
WIND_PROB    = 0.20
MAX_STEPS    = 50


class DroneRescueEnv(Env):
    """Gymnasium environment for an autonomous drone rescue mission.

    The drone operates on a 5x5 grid with obstacles, wind zones,
    danger zones, charging stations, and rescue targets.

    Observation:
        [row, column, battery_level]

    Action space:
        0 = Up, 1 = Down, 2 = Left, 3 = Right, 4 = Hover

    Termination:
        Battery reaches 0, all rescue targets collected,
        or maximum step limit (50) exceeded (truncation).
    """

    ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "HOVER"}

    SAFE_CELL         = SAFE
    START             = START
    WIND_ZONE         = WIND
    DANGER_ZONE       = DANGER
    CHARGING_STATION  = CHARGING
    RESCUE_TARGET     = RESCUE
    BLOCKED_CELL      = BLOCKED

    FULL_BATTERY_LEVEL           = MAX_BATTERY
    BATTERY_CONSUMPTION_PER_STEP = BATTERY_COST
    WIND_PROBABILITY             = WIND_PROB
    MAX_STEPS                    = MAX_STEPS

    OBSTACLE_MAP = GRID_MAP

    def __init__(self, render_mode=None, log_dir="logs"):
        """Initialise the environment, observation/action spaces, and state.

        Args:
            render_mode (str or None): "human" for Pygame rendering;
                None for headless operation.
            log_dir (str): Directory for logs and rendered video output.
        """
        super().__init__()

        self.grid = np.array(self.OBSTACLE_MAP, dtype=np.int32)
        self.action_space = Discrete(len(self.ACTION_NAMES))
        self.observation_space = Box(
            low=0,
            high=self.FULL_BATTERY_LEVEL,
            shape=(3,),
            dtype=np.int32,
        )

        self.state = (0, 0)
        self.battery_level = self.FULL_BATTERY_LEVEL
        self.step_count = 0
        self.rescue_target_state = {(0, 4): True, (4, 0): True}
        self.cumulative_reward = 0.0
        self.last_reward = 0.0
        self.render_mode = render_mode
        self.log_dir = log_dir
        self.renderer = None
        self.action_taken = None
        self.value_function = None
        self.policy = {}
        self.using_algorithm_policy = False
        self._update_policy_and_values()

        if self.render_mode is not None:
            self.renderer = self._create_renderer()

    # ----------------------------------------------------------
    # Core Gymnasium API
    # ----------------------------------------------------------

    def step(self, action):
        """Apply one action and return the Gymnasium 5-tuple.

        Args:
            action (int): Action to execute (0-4).

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        action = self._validate_action(action)
        x, y = self.state
        self.action_taken = self.ACTION_NAMES[action]

        if self.grid[x, y] == self.WIND_ZONE:
            if self.np_random.random() < self.WIND_PROBABILITY:
                action = int(self.np_random.choice([0, 1, 2, 3]))
                self.action_taken = f"{self.ACTION_NAMES.get(action)} (Wind)"

        next_x, next_y = self._get_candidate_position(action)
        self._apply_battery_cost(action)

        if self.grid[next_x, next_y] != self.BLOCKED_CELL:
            x, y = next_x, next_y

        if self.grid[x, y] == self.CHARGING_STATION:
            self.battery_level = self.FULL_BATTERY_LEVEL

        self.battery_level = max(self.battery_level, 0)
        self.state = (x, y)
        self.step_count += 1
        self.last_reward = self.calculate_reward()
        self.cumulative_reward += self.last_reward
        self._update_policy_and_values()
        done = self.check_done()
        return (self._get_obs(), self.last_reward,
                (done == 1), (done == 2), self._get_info())

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state.

        Args:
            seed (int, optional): RNG seed for reproducibility.
            options (dict, optional): Unused.

        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        self.state = (0, 0)
        self.grid = np.array(self.OBSTACLE_MAP, dtype=np.int32)
        self.battery_level = self.FULL_BATTERY_LEVEL
        self.step_count = 0
        self.rescue_target_state = {(0, 4): True, (4, 0): True}
        self.cumulative_reward = 0.0
        self.last_reward = 0.0
        self.action_taken = None
        self._update_policy_and_values()
        return self._get_obs(), self._get_info()

    def calculate_reward(self):
        """Calculate the reward for the current cell and battery state.

        Returns:
            float: Scalar reward.
        """
        x, y = self.state
        cell_type = self.grid[x, y]
        current_battery = self.battery_level

        if cell_type == self.RESCUE_TARGET:
            self.grid[x, y] = self.SAFE_CELL
            self.rescue_target_state[(x, y)] = False
            return 20

        if cell_type == self.DANGER_ZONE:
            return -10

        if current_battery == 0:
            return -20

        if cell_type == self.CHARGING_STATION:
            return 5

        return -1

    def check_done(self):
        """Check episode termination / truncation.

        Returns:
            int: 1 = terminated, 2 = truncated, 0 = still running.
        """
        current_rescue_targets = sum(self.rescue_target_state.values())
        if current_rescue_targets == 0:
            return 1
        if self.battery_level == 0:
            return 1
        if self.step_count >= self.MAX_STEPS:
            return 2
        return 0

    def render(self):
        """Render using the Pygame-based GridRenderer. No-op if headless."""
        if self.render_mode is None:
            return
        try:
            if self.renderer is None:
                self.renderer = self._create_renderer()
            self.renderer.render(
                grid=self.grid,
                state=self.state,
                battery_level=self.battery_level,
                step_count=self.step_count,
                cumulative_reward=self.cumulative_reward,
                rescue_target_state=self.rescue_target_state,
                action_taken=self.action_taken,
                value_function=self.value_function,
                policy=self.policy,
            )
        except RuntimeError as error:
            raise RuntimeError(
                f"Failed to render environment: {error}"
            ) from error

    def close(self):
        """Close the renderer and save captured video frames."""
        if self.render_mode is None:
            return
        if self.renderer is not None:
            self.renderer.save_video()
            self.renderer.close()

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _validate_action(self, action):
        """Normalise and validate the action."""
        if isinstance(action, np.integer):
            action = int(action)
        if not isinstance(action, int) or action not in self.ACTION_NAMES:
            valid = ", ".join(str(v) for v in self.ACTION_NAMES)
            raise ValueError(
                f"Invalid action {action!r}. Valid actions are: {valid}."
            )
        return action

    def _get_candidate_position(self, action):
        """Candidate (row, col) after movement, clipped to grid bounds."""
        x, y = self.state
        max_row = self.grid.shape[0] - 1
        max_col = self.grid.shape[1] - 1
        moves = {
            0: (max(x - 1, 0), y),
            1: (min(x + 1, max_row), y),
            2: (x, max(y - 1, 0)),
            3: (x, min(y + 1, max_col)),
            4: (x, y),
        }
        return moves[action]

    def _apply_battery_cost(self, action):
        """Deduct battery for movement; recharge if hovering on charger."""
        x, y = self.state
        if action == 4 and self.grid[x, y] == self.CHARGING_STATION:
            self.battery_level = min(
                self.battery_level + 2, self.FULL_BATTERY_LEVEL
            )
            return
        self.battery_level -= self.BATTERY_CONSUMPTION_PER_STEP

    def _update_policy_and_values(self):
        """Compute a heuristic value function and policy for rendering."""
        self.value_function = self._estimate_value_function()
        self.policy = self._estimate_direction_policy()
        self.using_algorithm_policy = False

    def _estimate_value_function(self):
        """Heuristic value function based on Manhattan distance to targets."""
        vf = np.zeros(self.grid.shape, dtype=np.float32)
        active = [t for t, a in self.rescue_target_state.items() if a]

        for row in range(self.grid.shape[0]):
            for col in range(self.grid.shape[1]):
                ct = self.grid[row, col]
                if ct == self.BLOCKED_CELL:
                    vf[row, col] = -1.0
                    continue
                if not active:
                    vf[row, col] = 1.0
                    continue
                d = min(abs(row - tr) + abs(col - tc) for tr, tc in active)
                mx = sum(self.grid.shape) - 2
                v = 1.0 - (d / mx)
                if ct == self.DANGER_ZONE:
                    v -= 0.35
                elif ct == self.CHARGING_STATION:
                    v += 0.2
                elif ct == self.RESCUE_TARGET:
                    v = 1.0
                vf[row, col] = np.clip(v, 0.0, 1.0)
        return vf

    def _estimate_direction_policy(self):
        """Greedy direction policy toward nearest active rescue target."""
        policy = {}
        active = [t for t, a in self.rescue_target_state.items() if a]
        if not active:
            return policy
        for row in range(self.grid.shape[0]):
            for col in range(self.grid.shape[1]):
                if self.grid[row, col] == self.BLOCKED_CELL:
                    continue
                if (row, col) in active:
                    policy[(row, col)] = 4
                    continue
                policy[(row, col)] = self._best_action_toward_target(
                    row, col, active
                )
        return policy

    def _best_action_toward_target(self, row, col, active_targets):
        """Return the one-step greedy action toward the nearest target."""
        cands = {
            0: (max(row - 1, 0), col),
            1: (min(row + 1, self.grid.shape[0] - 1), col),
            2: (row, max(col - 1, 0)),
            3: (row, min(col + 1, self.grid.shape[1] - 1)),
            4: (row, col),
        }
        best_a, best_d = 4, float("inf")
        for a, (nr, nc) in cands.items():
            if self.grid[nr, nc] == self.BLOCKED_CELL:
                continue
            d = min(abs(nr - tr) + abs(nc - tc) for tr, tc in active_targets)
            if d < best_d:
                best_a, best_d = a, d
        return best_a

    def _get_obs(self):
        """Current observation: [row, col, battery]."""
        return np.array(
            [self.state[0], self.state[1], self.battery_level],
            dtype=np.int32,
        )

    def _get_info(self):
        """Diagnostic info dictionary for logging and rendering."""
        return {
            "state": self.state,
            "step_count": self.step_count,
            "battery_level": self.battery_level,
            "rescue_target_state": self.rescue_target_state,
            "action_taken": self.action_taken,
            "value_function": self.value_function,
            "policy": self.policy,
            "using_algorithm_policy": self.using_algorithm_policy,
        }

    def _create_renderer(self):
        """Lazily create the Pygame renderer from the support folder."""
        dp_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "Drone_Rescue_DP",
        )
        if dp_dir not in sys.path:
            sys.path.insert(0, dp_dir)
        try:
            from renderer import GridRenderer
        except ImportError:
            raise ImportError(
                "GridRenderer not found. Ensure renderer.py is in "
                "Drone_Rescue_DP/."
            )
        return GridRenderer(rows=5, cols=5, fps=30, log_dir=self.log_dir)
