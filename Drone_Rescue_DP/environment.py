"""Drone rescue Gymnasium environment.

This module contains the core simulation rules for the autonomous drone rescue
task. The environment models a fixed grid with obstacles, wind, danger zones,
charging stations, and rescue targets. It uses the Gymnasium API so the same
environment can be used by manual scripts, dynamic-programming routines, or
reinforcement-learning agents.

Observation:
    A compact vector in the form ``[row, column, battery_level]``.

Action space:
    ``0`` = up, ``1`` = down, ``2`` = left, ``3`` = right, ``4`` = hover.

Termination:
    The episode terminates when all rescue targets are collected or when the
    battery reaches zero. It is truncated when ``MAX_STEPS`` is reached.
"""

from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import numpy as np


class DroneRescueEnv(Env):
    """Gymnasium environment for a drone rescue mission on a fixed grid.

    The environment intentionally keeps the observation small: the agent sees
    its current location and battery level rather than the entire grid. The grid
    is still stored internally so reward calculation, collision handling, and
    rendering can use the full map.

    Grid legend:
        ``-1``: blocked cell
        ``0``: safe traversable cell
        ``1``: start cell
        ``2``: wind zone
        ``3``: danger zone
        ``4``: charging station
        ``5``: rescue target
    """

    # Action ids are kept as integers so they work naturally with Gymnasium's
    # Discrete action space and with simple planning algorithms.
    ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "HOVER"}

    # Cell-type constants make the reward and transition rules readable.
    SAFE_CELL = 0
    START = 1
    WIND_ZONE = 2
    DANGER_ZONE = 3
    CHARGING_STATION = 4
    RESCUE_TARGET = 5
    BLOCKED_CELL = -1

    FULL_BATTERY_LEVEL = 15
    BATTERY_CONSUMPTION_PER_STEP = 1
    WIND_PROBABILITY = 0.2
    MAX_STEPS = 50

    # Fixed map for the current rescue scenario. Multiple rescue targets are
    # allowed; each target is marked safe after it is rescued.
    OBSTACLE_MAP = [
        [1,  0,  2,  0,  5],
        [0, -1,  3,  0,  2],
        [0,  0,  4,  0,  0],
        [3,  0, -1,  0,  0],
        [5,  0,  0,  3,  0],
    ]

    def __init__(self, render_mode=None, log_dir="logs"):
        """Initialize the DroneRescueEnv environment.

        The observation is ``[row, column, battery_level]``. The grid itself is
        static except for rescue targets, which are marked safe after rescue.
        The renderer is created lazily, so non-visual training or tests can use
        the environment without importing or initializing Pygame.

        Attributes:
            grid: The environment grid.
            action_space (Discrete): Action space with 5 discrete actions.
            observation_space (Box): Observation vector with row, column, and battery.
            state (tuple): Current agent position, initialized to (0, 0).
            battery_level (int): Current battery level of the drone.
            step_count (int): Counter for the number of steps taken in the episode.
            rescue_target_state (dict): Dictionary to track the state of rescue targets.
            cumulative_reward (float): Total reward accumulated during the episode.
            render_mode (str): Mode for rendering the environment.
            log_dir (str): Directory for logging simulation data.
        """
        super().__init__()

        # Gymnasium spaces describe what an external agent is allowed to send
        # and what shape/type it can expect to receive from the environment.
        self.grid = np.array(self.OBSTACLE_MAP, dtype=np.int32)
        self.action_space = Discrete(len(self.ACTION_NAMES))
        self.observation_space = Box(
            low=0,
            high=self.FULL_BATTERY_LEVEL,
            shape=(3,),
            dtype=np.int32,
        )

        self.state = (0, 0)  # Starting position
        self.battery_level = self.FULL_BATTERY_LEVEL
        self.step_count = 0
        self.rescue_target_state = {(0, 4): True, (4, 0): True}
        self.cumulative_reward = 0.0
        self.last_reward = 0.0
        self.render_mode = render_mode
        self.log_dir = log_dir
        self.renderer = None
        self.action_taken = None

        if self.render_mode is not None:
            self.renderer = self._create_renderer()

    def step(self, action):
        """Apply one action and return the standard Gymnasium step output.

        This method is the main transition function. It validates the action,
        applies wind if needed, computes the candidate position, updates the
        battery, prevents blocked-cell movement, calculates reward, and reports
        whether the episode has ended.

        Args:
            action (int): The action to take (0: Up, 1: Down, 2: Left, 3: Right, 4: Hover).

        Returns:
            observation (np.ndarray): The state observation after the action.
            reward (float): The reward received for the action.
            terminated (bool): Whether the episode terminated.
            truncated (bool): Whether the episode was truncated.
            info (dict): Additional diagnostic information.

        """
        action = self._validate_action(action)
        x, y = self.state
        self.action_taken = self.ACTION_NAMES[action]

        # Wind zones introduce stochasticity by occasionally replacing the
        # intended action with a random cardinal move.
        if self.grid[x, y] == self.WIND_ZONE:
            if np.random.rand() < self.WIND_PROBABILITY:
                action = int(np.random.choice([0, 1, 2, 3]))
                self.action_taken = f"{self.ACTION_NAMES.get(action)} (Wind)"

        next_x, next_y = self._get_candidate_position(action)
        self._apply_battery_cost(action)

        # Blocked cells act like walls. The action still consumes battery, but
        # the drone remains in its previous valid location.
        if self.grid[next_x, next_y] != self.BLOCKED_CELL:
            x, y = next_x, next_y

        # Reaching a charging station immediately restores the battery. Hovering
        # on a charging station is also handled in _apply_battery_cost.
        if self.grid[x, y] == self.CHARGING_STATION:
            self.battery_level = self.FULL_BATTERY_LEVEL

        # Keep the battery bounded at zero so downstream logic does not need to
        # handle negative battery values.
        self.battery_level = max(self.battery_level, 0)
        self.state = (x, y)
        self.step_count += 1
        self.last_reward = self.calculate_reward()
        self.cumulative_reward += self.last_reward
        done = self.check_done()
        return self._get_obs(), self.last_reward, (done == 1), (done == 2), self._get_info()

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state.

        Gymnasium calls this at the beginning of each episode. The full grid is
        reconstructed so rescued targets from a previous episode reappear.

        Args:
            seed (int, optional): Optional seed for random number generation.
            options (dict, optional): Additional reset options.

        Returns:
            observation (np.ndarray): The initial observation after reset.
            info (dict): Auxiliary information about the reset state.
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
        return self._get_obs(), self._get_info()

    def _validate_action(self, action):
        """Validate and normalize an action before applying it.

        Args:
            action: Candidate action supplied by an agent or script.

        Returns:
            int: A normalized Python integer action id.

        Raises:
            ValueError: If the action is not an integer in ``ACTION_NAMES``.
        """
        if isinstance(action, np.integer):
            action = int(action)

        if not isinstance(action, int) or action not in self.ACTION_NAMES:
            valid_actions = ", ".join(str(value) for value in self.ACTION_NAMES)
            raise ValueError(f"Invalid action {action!r}. Valid actions are: {valid_actions}.")

        return action

    def _get_candidate_position(self, action):
        """Return the next grid position requested by an action.

        The returned position is clipped to the grid boundaries. Blocked-cell
        checks happen later in ``step`` because the candidate position is still
        useful for deciding whether movement should be allowed.

        Args:
            action (int): Valid action id.

        Returns:
            tuple[int, int]: Candidate ``(row, column)`` position.
        """
        x, y = self.state
        max_row, max_col = self.grid.shape[0] - 1, self.grid.shape[1] - 1

        moves = {
            0: (max(x - 1, 0), y),
            1: (min(x + 1, max_row), y),
            2: (x, max(y - 1, 0)),
            3: (x, min(y + 1, max_col)),
            4: (x, y),
        }
        return moves[action]

    def _apply_battery_cost(self, action):
        """Update battery level after a movement or hover action.

        Movement always costs one unit of battery. Hovering usually costs one
        unit as well, but hovering on a charging station slowly recharges the
        drone without exceeding ``FULL_BATTERY_LEVEL``.

        Args:
            action (int): Valid action id.
        """
        x, y = self.state

        if action == 4 and self.grid[x, y] == self.CHARGING_STATION:
            self.battery_level = min(self.battery_level + 2, self.FULL_BATTERY_LEVEL)
            return

        self.battery_level -= self.BATTERY_CONSUMPTION_PER_STEP

    def calculate_reward(self):
        """Calculate the reward for the current state.

        Rewards are intentionally simple and sparse:
        rescuing a target is highly positive, danger and battery depletion are
        negative, charging is mildly positive, and ordinary movement receives a
        small step penalty to encourage shorter routes.

        Returns:
            reward (float): The calculated reward based on current cell and battery.
        """
        x, y = self.state
        cell_type = self.grid[x, y]
        current_battery = self.battery_level

        if cell_type == self.RESCUE_TARGET:
            # Mark the target as rescued so it cannot be counted again.
            self.grid[x, y] = self.SAFE_CELL
            self.rescue_target_state[(x, y)] = False  # Update rescue target tracking.
            return 20  # High reward for rescuing the target.

        if cell_type == self.DANGER_ZONE:
            return -10  # Penalty for entering a danger zone.

        if current_battery == 0:
            return -20  # Large penalty for running out of battery.

        if cell_type == self.CHARGING_STATION:
            return 5  # Reward for reaching a charging station.

        return -1  # Step penalty to encourage efficient behavior.

    def check_done(self):
        """Check if the episode has ended.

        The return value is an internal status code that is converted into
        Gymnasium's ``terminated`` and ``truncated`` booleans by ``step``.

        Returns:
            int: 1 for termination, 2 for truncation, and 0 while running.
        """
        current_rescue_targets = sum(self.rescue_target_state.values())
        if current_rescue_targets == 0:
            return 1  # Episode ends when all rescue targets are rescued

        if self.battery_level == 0:
            return 1  # Episode ends when battery runs out

        if self.step_count >= self.MAX_STEPS:
            return 2  # Episode ends when maximum steps are reached

        return 0

    def render(self):
        """Render the environment using the configured renderer.

        Rendering is optional. When ``render_mode`` is ``None``, this method is a
        no-op so experiments can run headlessly. Runtime rendering failures are
        wrapped with a clearer environment-level error message.

        Returns:
            None
        """
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
            )
        except RuntimeError as error:
            raise RuntimeError(f"Failed to render environment: {error}") from error

    def _create_renderer(self):
        """Create the renderer only when visual output is requested.

        Importing the renderer lazily avoids requiring Pygame/ImageIO during
        non-rendering tests, training runs, or server-side batch experiments.

        Returns:
            GridRenderer: Renderer configured for this environment instance.
        """
        try:
            from .renderer import GridRenderer
        except ImportError:
            from renderer import GridRenderer

        return GridRenderer(rows=5, cols=5, fps=30, log_dir=self.log_dir)

    def _get_obs(self):
        """Return the current observation vector.

        The observation intentionally excludes the full grid map. Algorithms
        that need map knowledge can access ``env.grid`` directly, while learning
        agents can work from this compact state vector.

        Returns:
            np.ndarray: The current observation [x, y, battery_level].
        """
        return np.array([self.state[0], self.state[1], self.battery_level], dtype=np.int32)

    def _get_info(self):
        """Return auxiliary environment information for diagnostics.

        The ``info`` dictionary is meant for logging, debugging, and rendering.
        Agents should generally learn from the observation and reward, not from
        ``info`` values that may expose extra environment details.

        Returns:
            dict: Diagnostic information including state and action metadata.
        """
        return {
            "state": self.state,
            "step_count": self.step_count,
            "battery_level": self.battery_level,
            "rescue_target_state": self.rescue_target_state,
            "action_taken": self.action_taken,
        }

    def close(self):
        """Close the renderer and finalize output.

        When rendering is enabled, the renderer saves the captured video before
        releasing Pygame resources. In headless mode this method is a no-op.
        """
        if self.render_mode is None:
            return

        if self.renderer is not None:
            self.renderer.save_video()
            self.renderer.close()
