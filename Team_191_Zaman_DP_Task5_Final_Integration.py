"""
=============================================================
BIRLA INSTITUTE OF TECHNOLOGY AND SCIENCE, PILANI
WORK INTEGRATED LEARNING PROGRAMMES DIVISION

Deep Reinforcement Learning - Lab Assignment 1
Part #2: Dynamic Programming
Team Number: 191

Contributor: Md M Zaman

Task Implemented:
  DP - State value analysis + DP scalability discussion + final integration

This file is intentionally self-contained so it can be submitted as the
Team 191 DP final-integration component. It reuses the drone rescue MDP
structure developed by the team and adds the missing Task 5 analysis:

  1. Compute V*(s) with Value Iteration.
  2. Visualize meaningful state-value slices as heatmaps.
  3. Simulate the optimal policy path for final integration evidence.
  4. Print a scalability discussion for larger grids, extra targets,
     and dynamic weather.
=============================================================
"""

import itertools
import os
import socket
import subprocess
import time
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------
# Section 1: Assignment and environment constants
# ------------------------------------------------------------

TEAM_NUMBER = 191

# Last digit of 191 is 1, so the assignment configuration uses
# a 5x5 grid, 15 battery units, and 20% wind deviation.
GRID_MAP = [
    [1, 0, 2, 0, 5],
    [0, -1, 3, 0, 2],
    [0, 0, 4, 0, 0],
    [3, 0, -1, 0, 0],
    [5, 0, 0, 3, 0],
]

SAFE = 0
START = 1
WIND = 2
DANGER = 3
CHARGING = 4
RESCUE = 5
BLOCKED = -1

CELL_LABEL = {
    SAFE: "F",
    START: "S",
    WIND: "W",
    DANGER: "D",
    CHARGING: "C",
    RESCUE: "R",
    BLOCKED: "X",
}

UP, DOWN, LEFT, RIGHT, HOVER = 0, 1, 2, 3, 4
ACTIONS = [UP, DOWN, LEFT, RIGHT, HOVER]
ACTION_NAME = {
    UP: "UP",
    DOWN: "DOWN",
    LEFT: "LEFT",
    RIGHT: "RIGHT",
    HOVER: "HOVER",
}
ACTION_ARROW = {
    UP: "^",
    DOWN: "v",
    LEFT: "<",
    RIGHT: ">",
    HOVER: "o",
}

ROWS = 5
COLS = 5
MAX_BATTERY = 15
BATTERY_COST = 1
WIND_PROB = 0.20
GAMMA = 0.99
THETA = 1e-3
MAX_POLICY_STEPS = 50

R_RESCUE = 20
R_DANGER = -10
R_DEAD = -20
R_STEP = -1
R_CHARGE = 5

START_POS = (0, 0)
RESCUE_POSITIONS = [(0, 4), (4, 0)]
CHARGING_POSITIONS = [(2, 2)]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_execution_info():
    """Return timestamp and VM/host information required by the assignment."""
    try:
        timestamp = subprocess.check_output(["date"], stderr=subprocess.DEVNULL)
        timestamp = timestamp.decode("utf-8", errors="replace").strip()
    except Exception:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        vm_id = subprocess.check_output(["hostname"], stderr=subprocess.DEVNULL)
        vm_id = vm_id.decode("utf-8", errors="replace").strip()
    except Exception:
        vm_id = socket.gethostname()

    return timestamp, vm_id


@dataclass(frozen=True)
class Transition:
    """One possible transition outcome for a state-action pair."""

    probability: float
    next_state: tuple
    reward: float


class DroneRescueMDP:
    """Finite MDP model for the autonomous drone rescue task.

    State representation:
        (row, col, battery, rescued_tuple, charger_visited)

    The state includes the drone position, remaining battery, which
    rescue targets have already been collected, and whether the charger
    reward has already been used. Including rescue and charger memory
    keeps the process Markovian.
    """

    def __init__(self):
        self.grid = np.array(GRID_MAP, dtype=np.int32)
        self.valid_positions = [
            (r, c)
            for r in range(ROWS)
            for c in range(COLS)
            if self.grid[r, c] != BLOCKED
        ]
        self.states = self._enumerate_states()
        self.state_to_index = {state: idx for idx, state in enumerate(self.states)}
        self.terminal_states = {
            state for state in self.states if state[2] == 0 or all(state[3])
        }

    def _enumerate_states(self):
        """Enumerate position x battery x rescued-combo x charger flag."""
        rescued_combinations = list(
            itertools.product([False, True], repeat=len(RESCUE_POSITIONS))
        )
        return [
            (r, c, battery, rescued, charger_visited)
            for (r, c) in self.valid_positions
            for battery in range(MAX_BATTERY + 1)
            for rescued in rescued_combinations
            for charger_visited in [False, True]
        ]

    def cell_type(self, row, col, rescued):
        """Return effective cell type after removing rescued targets."""
        cell = int(self.grid[row, col])
        if cell == RESCUE:
            for idx, rescue_pos in enumerate(RESCUE_POSITIONS):
                if (row, col) == rescue_pos and rescued[idx]:
                    return SAFE
        return cell

    def valid_actions(self, state):
        """Return actions that are valid from the current state."""
        if state in self.terminal_states:
            return []

        row, col, _, _, _ = state
        valid = [HOVER]
        for action in [UP, DOWN, LEFT, RIGHT]:
            nr, nc = self._move(row, col, action)
            if (nr, nc) != (row, col) and self.grid[nr, nc] != BLOCKED:
                valid.append(action)
        return valid

    @staticmethod
    def _move(row, col, action):
        """Return the clipped candidate position for an action."""
        if action == UP:
            row -= 1
        elif action == DOWN:
            row += 1
        elif action == LEFT:
            col -= 1
        elif action == RIGHT:
            col += 1

        row = min(max(row, 0), ROWS - 1)
        col = min(max(col, 0), COLS - 1)
        return row, col

    def _apply_action(self, state, action):
        """Apply one deterministic action and return next state plus reward."""
        row, col, battery, rescued, charger_visited = state

        if state in self.terminal_states:
            return state, 0.0

        next_row, next_col = self._move(row, col, action)
        if self.grid[next_row, next_col] == BLOCKED:
            next_row, next_col = row, col

        next_battery = max(battery - BATTERY_COST, 0)
        next_rescued = list(rescued)
        next_charger_visited = charger_visited

        reward = R_STEP
        effective_cell = self.cell_type(next_row, next_col, rescued)

        if next_battery == 0:
            reward = R_DEAD
        elif effective_cell == DANGER:
            reward = R_DANGER
        elif effective_cell == RESCUE:
            reward = R_RESCUE
            for idx, rescue_pos in enumerate(RESCUE_POSITIONS):
                if (next_row, next_col) == rescue_pos:
                    next_rescued[idx] = True
        elif effective_cell == CHARGING:
            if not charger_visited:
                reward = R_CHARGE
            next_battery = MAX_BATTERY
            next_charger_visited = True

        next_state = (
            next_row,
            next_col,
            next_battery,
            tuple(next_rescued),
            next_charger_visited,
        )
        return next_state, float(reward)

    def transitions(self, state, action):
        """Return stochastic transition outcomes for state and action.

        Wind zones deviate the action with probability WIND_PROB. The
        deviation is split equally across the four movement actions.
        Duplicate outcomes are aggregated into a single probability.
        """
        row, col, _, rescued, _ = state
        if self.cell_type(row, col, rescued) == WIND:
            action_probs = {action: 1.0 - WIND_PROB}
            for wind_action in [UP, DOWN, LEFT, RIGHT]:
                action_probs[wind_action] = action_probs.get(wind_action, 0.0)
                action_probs[wind_action] += WIND_PROB / 4.0
        else:
            action_probs = {action: 1.0}

        aggregated = {}
        for actual_action, probability in action_probs.items():
            next_state, reward = self._apply_action(state, actual_action)
            key = (next_state, reward)
            aggregated[key] = aggregated.get(key, 0.0) + probability

        return [
            Transition(probability=prob, next_state=next_state, reward=reward)
            for (next_state, reward), prob in aggregated.items()
        ]


def value_iteration(mdp):
    """Compute optimal values and policy using Value Iteration."""
    values = {state: 0.0 for state in mdp.states}
    deltas = []
    start_time = time.perf_counter()

    iteration = 0
    while True:
        iteration += 1
        delta = 0.0
        new_values = values.copy()

        for state in mdp.states:
            if state in mdp.terminal_states:
                continue

            action_returns = []
            for action in mdp.valid_actions(state):
                expected_return = sum(
                    transition.probability
                    * (transition.reward + GAMMA * values[transition.next_state])
                    for transition in mdp.transitions(state, action)
                )
                action_returns.append(expected_return)

            best_value = max(action_returns)
            new_values[state] = best_value
            delta = max(delta, abs(best_value - values[state]))

        values = new_values
        deltas.append(delta)

        if delta < THETA:
            break

    runtime = time.perf_counter() - start_time
    policy = extract_policy(mdp, values)
    return values, policy, deltas, runtime


def extract_policy(mdp, values):
    """Derive a greedy optimal policy from the converged value function."""
    policy = {}
    for state in mdp.states:
        if state in mdp.terminal_states:
            policy[state] = None
            continue

        best_action = None
        best_return = -float("inf")
        for action in mdp.valid_actions(state):
            expected_return = sum(
                transition.probability
                * (transition.reward + GAMMA * values[transition.next_state])
                for transition in mdp.transitions(state, action)
            )
            if expected_return > best_return:
                best_return = expected_return
                best_action = action

        policy[state] = best_action
    return policy


def build_value_grid(mdp, values, battery, rescued, charger_visited=False):
    """Create a 5x5 grid for a fixed battery/rescue state slice."""
    value_grid = np.full((ROWS, COLS), np.nan)
    action_grid = [["" for _ in range(COLS)] for _ in range(ROWS)]

    for row, col in mdp.valid_positions:
        state = (row, col, battery, tuple(rescued), charger_visited)
        value_grid[row, col] = values[state]

    return value_grid, action_grid


def plot_state_value_heatmaps(mdp, values, policy):
    """Plot meaningful V*(s) slices for Task 5 state-value analysis."""
    slices = [
        (15, (False, False), False, "Battery 15, no rescue"),
        (7, (False, False), False, "Battery 7, no rescue"),
        (3, (False, False), False, "Battery 3, no rescue"),
        (1, (False, False), False, "Battery 1, no rescue"),
        (15, (True, False), False, "T0 rescued, battery 15"),
        (15, (False, True), False, "T1 rescued, battery 15"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    all_values = []
    grids = []

    for battery, rescued, charger_visited, _ in slices:
        grid, _ = build_value_grid(mdp, values, battery, rescued, charger_visited)
        grids.append(grid)
        all_values.extend(grid[~np.isnan(grid)].ravel())

    vmin = min(all_values)
    vmax = max(all_values)

    for ax, grid, slice_info in zip(axes.ravel(), grids, slices):
        battery, rescued, charger_visited, title = slice_info
        im = ax.imshow(grid, cmap="RdYlGn", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(range(COLS))
        ax.set_yticks(range(ROWS))

        for row in range(ROWS):
            for col in range(COLS):
                cell = GRID_MAP[row][col]
                if cell == BLOCKED:
                    ax.text(col, row, "X", ha="center", va="center", fontweight="bold")
                    continue

                state = (row, col, battery, tuple(rescued), charger_visited)
                action = policy.get(state)
                arrow = ACTION_ARROW.get(action, ".")
                value = values[state]
                label = f"{CELL_LABEL[cell]}\n{value:.1f}\n{arrow}"
                ax.text(col, row, label, ha="center", va="center", fontsize=8)

        ax.grid(color="black", linewidth=0.4)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="Optimal value V*(s)")
    fig.suptitle(
        "Team 191 Task 5: State-Value Heatmaps for Drone Rescue DP",
        fontsize=14,
        fontweight="bold",
    )

    output_path = os.path.join(SCRIPT_DIR, "Team_191_Task5_State_Value_Heatmaps.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def most_likely_next_state(mdp, state, action):
    """Pick the highest-probability transition for readable path tracing."""
    candidates = mdp.transitions(state, action)
    candidates = sorted(candidates, key=lambda item: item.probability, reverse=True)
    return candidates[0].next_state, candidates[0].reward


def simulate_policy_path(mdp, policy):
    """Follow the optimal policy from the start state for integration evidence."""
    state = (START_POS[0], START_POS[1], MAX_BATTERY, (False, False), False)
    path = [(state, None, 0.0)]
    total_reward = 0.0

    for _ in range(MAX_POLICY_STEPS):
        if state in mdp.terminal_states:
            break

        action = policy[state]
        next_state, reward = most_likely_next_state(mdp, state, action)
        total_reward += reward
        path.append((next_state, action, reward))
        state = next_state

        if state in mdp.terminal_states:
            break

    return path, total_reward


def plot_optimal_path(path):
    """Plot the deterministic trace of the optimal policy."""
    fig, ax = plt.subplots(figsize=(6, 6))
    display_grid = np.array(GRID_MAP, dtype=float)
    display_grid[display_grid == BLOCKED] = np.nan
    ax.imshow(display_grid, cmap="Pastel2", alpha=0.75)

    positions = [(state[0], state[1]) for state, _, _ in path]
    rows = [pos[0] for pos in positions]
    cols = [pos[1] for pos in positions]

    ax.plot(cols, rows, color="#1f77b4", linewidth=2, marker="o")
    for idx, (row, col) in enumerate(positions):
        ax.text(col, row, str(idx), ha="center", va="center", fontsize=8)

    for row in range(ROWS):
        for col in range(COLS):
            label = CELL_LABEL[GRID_MAP[row][col]]
            ax.text(col, row + 0.32, label, ha="center", va="center", fontsize=8)

    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.set_title("Team 191 Task 5: Optimal Policy Path")
    ax.grid(color="black", linewidth=0.4)

    output_path = os.path.join(SCRIPT_DIR, "Team_191_Task5_Optimal_Path.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def print_state_value_analysis(values):
    """Print written observations required for the state-value analysis mark."""
    start_state = (0, 0, MAX_BATTERY, (False, False), False)
    charger_state = (2, 2, 7, (False, False), False)
    low_battery_state = (0, 0, 1, (False, False), False)
    after_t0_state = (0, 4, MAX_BATTERY, (True, False), False)

    print("\nTASK 5A: STATE-VALUE ANALYSIS")
    print("-" * 60)
    print(f"V*(start, full battery, no rescues) = {values[start_state]:.3f}")
    print(f"V*(charger, battery 7, no rescues) = {values[charger_state]:.3f}")
    print(f"V*(start, battery 1, no rescues) = {values[low_battery_state]:.3f}")
    print(f"V*(T0 cell after T0 rescued)       = {values[after_t0_state]:.3f}")
    print()
    print("Observed patterns:")
    print("1. High-battery states near rescue targets have the highest values.")
    print("2. Low-battery states become negative unless the drone is close to a")
    print("   rescue target or the charging station.")
    print("3. After one target is rescued, the value landscape shifts toward")
    print("   the remaining target, proving that the rescued tuple is necessary")
    print("   in the state representation.")
    print("4. Danger and blocked regions reduce nearby values because they either")
    print("   create direct penalties or restrict efficient movement.")


def print_scalability_discussion(mdp):
    """Print the DP scalability discussion required by Task 5."""
    valid_positions_5x5 = len(mdp.valid_positions)
    states_5x5 = len(mdp.states)

    # Approximate 10x10 state count with all cells valid. This is a clear
    # upper-bound style estimate for the assignment discussion.
    states_10x10_two_targets = 100 * (MAX_BATTERY + 1) * (2 ** 2) * 2
    states_10x10_four_targets = 100 * (MAX_BATTERY + 1) * (2 ** 4) * 2

    # If weather is dynamic, add a simple weather-mode variable. Three
    # modes are enough to show the multiplicative growth clearly.
    weather_modes = 3
    states_dynamic_weather = states_10x10_four_targets * weather_modes

    print("\nTASK 5B: DP SCALABILITY DISCUSSION")
    print("-" * 60)
    print(f"Current valid grid positions       : {valid_positions_5x5}")
    print(f"Current exact DP state count       : {states_5x5}")
    print(f"10x10 grid, 2 targets estimate     : {states_10x10_two_targets}")
    print(f"10x10 grid, 4 targets estimate     : {states_10x10_four_targets}")
    print(f"With 3 dynamic weather modes       : {states_dynamic_weather}")
    print()
    print("Discussion:")
    print("Dynamic Programming is exact and interpretable for this 5x5 problem,")
    print("but it scales poorly because the state count grows multiplicatively:")
    print("positions x battery levels x rescue combinations x charger/weather")
    print("memory. Adding rescue targets is especially expensive because the")
    print("rescued-target component grows as 2^targets. With larger maps and")
    print("dynamic weather, full transition enumeration becomes slow and memory")
    print("heavy. Deep RL methods such as DQN or actor-critic algorithms can help")
    print("by learning value/policy approximations from sampled experience instead")
    print("of storing every state explicitly, which is closer to real autonomous")
    print("drone systems where wind, obstacles, and victims can change online.")


def print_final_integration_summary(path, total_reward, heatmap_path, route_path):
    """Print final integration notes connecting all team work."""
    final_state = path[-1][0]
    rescued = final_state[3]

    print("\nTASK 5C: FINAL INTEGRATION SUMMARY")
    print("-" * 60)
    print("Integrated deliverables for Team 191:")
    print("1. Aamna's MAB file covers dataset design and immediate exploitation.")
    print("2. Team MAB work can compare exploitation, epsilon-greedy, and UCB1.")
    print("3. This DP file completes the drone-rescue final piece by adding")
    print("   state-value analysis, scalability discussion, and optimal-path")
    print("   integration evidence.")
    print()
    print(f"Optimal policy trace length : {len(path) - 1} actions")
    print(f"Most-likely trace reward    : {total_reward:.1f}")
    print(f"Targets rescued             : {rescued}")
    print(f"State-value heatmap saved   : {heatmap_path}")
    print(f"Optimal path plot saved     : {route_path}")

    print("\nOptimal policy trace:")
    for step, (state, action, reward) in enumerate(path):
        row, col, battery, rescued_tuple, charger_visited = state
        action_text = "START" if action is None else ACTION_NAME[action]
        print(
            f"Step {step:02d}: pos=({row},{col}) battery={battery:02d} "
            f"rescued={rescued_tuple} charger={charger_visited} "
            f"action={action_text:<5} reward={reward:>5.1f}"
        )


def main():
    """Run the Task 5 DP integration workflow."""
    timestamp, vm_id = get_execution_info()
    print("=" * 60)
    print("TEAM 191 - DP TASK 5 FINAL INTEGRATION")
    print("=" * 60)
    print(f"Execution timestamp : {timestamp}")
    print(f"Virtual machine ID  : {vm_id}")
    print(f"Team number         : {TEAM_NUMBER}")
    print(f"Discount gamma      : {GAMMA}")
    print(f"Stopping theta      : {THETA}")

    mdp = DroneRescueMDP()
    print("\nMDP state representation:")
    print("(row, col, battery, rescued_tuple, charger_visited)")
    print(f"Valid positions : {len(mdp.valid_positions)}")
    print(f"Total states    : {len(mdp.states)}")
    print(f"Terminal states : {len(mdp.terminal_states)}")

    values, policy, deltas, runtime = value_iteration(mdp)
    print("\nVALUE ITERATION RESULT")
    print("-" * 60)
    print(f"Convergence iterations : {len(deltas)}")
    print(f"Runtime seconds        : {runtime:.4f}")
    print(f"Final delta/error      : {deltas[-1]:.6f}")

    heatmap_path = plot_state_value_heatmaps(mdp, values, policy)
    path, total_reward = simulate_policy_path(mdp, policy)
    route_path = plot_optimal_path(path)

    print_state_value_analysis(values)
    print_scalability_discussion(mdp)
    print_final_integration_summary(path, total_reward, heatmap_path, route_path)

    print("\nSubmission status: Team 191 DP Task 5 complete.")


if __name__ == "__main__":
    main()
