"""
TEAM 181 — DP: Autonomous Drone Rescue Using Dynamic Programming
Course: Deep Reinforcement Learning — Lab Assignment 1, Part 2
Team: 181  |  Grid: 5x5  |  Battery: 15  |  Wind: 20%  |  γ=0.99  θ=10⁻³

Run:  python3 "TEAM_181 - DP.py"
See:  README.md for full file structure and expected outcomes.
"""

# ================================================================
# Section 1 — Imports, VM Info, Shared Constants
# ================================================================

import numpy as np
import time
import itertools
import os
import sys
import subprocess
import logging
from datetime import datetime as dt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    _HAS_GYMNASIUM = True
    from gymnasium_env import DroneRescueEnv
except ImportError:
    _HAS_GYMNASIUM = False
    DroneRescueEnv = None

def get_vm_info():
    """Fetch the VM timestamp and hostname for assignment compliance.

    The assignment mandates that every submission prints the execution
    timestamp and virtual-machine identifier to confirm the code was
    run inside the virtual lab.

    Returns:
        tuple: (timestamp_string, vm_id_string)
    """
    try:
        timestamp = subprocess.check_output(
            ["date"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        timestamp = "Unavailable"
    try:
        vm_id = subprocess.check_output(
            ["hostname"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        vm_id = "Unavailable"
    return timestamp, vm_id


# ---- Group & Environment Constants ----
G = 181

GRID_MAP = [
    [1,  0,  2,  0,  5],   # Row 0: S  F  W  F  R
    [0, -1,  3,  0,  2],   # Row 1: F  X  D  F  W
    [0,  0,  4,  0,  0],   # Row 2: F  F  C  F  F
    [3,  0, -1,  0,  0],   # Row 3: D  F  X  F  F
    [5,  0,  0,  3,  0],   # Row 4: R  F  F  D  F
]

ROWS, COLS = 5, 5

# Cell-type constants
SAFE       =  0
START      =  1
WIND       =  2
DANGER     =  3
CHARGING   =  4
RESCUE     =  5
BLOCKED    = -1

CELL_SYMBOL = {
    SAFE: "F", START: "S", WIND: "W", DANGER: "D",
    CHARGING: "C", RESCUE: "R", BLOCKED: "X",
}

# Actions
UP, DOWN, LEFT, RIGHT, HOVER = 0, 1, 2, 3, 4
ACTIONS      = [UP, DOWN, LEFT, RIGHT, HOVER]
ACTION_NAME  = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "HOVER"}
ACTION_ARROW = {0: "↑",  1: "↓",   2: "←",    3: "→",     4: "●"}

# Battery
MAX_BATTERY  = 15
BATTERY_COST = 1

# Wind
WIND_PROB = 0.20

# Rewards (from assignment PDF, Section "Rewards")
R_RESCUE  =  20
R_DANGER  = -10
R_DEAD    = -20
R_STEP    =  -1
R_CHARGE  =   5   # One-time entry reward (see note below)

# Rescue-target positions (fixed by the grid layout)
RESCUE_POS  = [(0, 4), (4, 0)]
NUM_TARGETS = len(RESCUE_POS)

# DP hyper-parameters
GAMMA = 0.99
THETA = 1e-3

MAX_STEPS = 50


# ================================================================
# Section 2 — Expected Outcome 2: MDP Model
# ================================================================
# Full state-space enumeration and transition function for the
# Drone Rescue MDP.  The transition model mirrors DroneRescueEnv.step()
# so that the DP solution is consistent with the simulation.
# ================================================================

class DroneRescueMDP:
    """Markov Decision Process for the drone rescue grid.

    State representation:  ``(row, col, battery, rescued, charged)``

    * ``row``, ``col``   — drone position on the 5 × 5 grid
    * ``battery``        — integer in [0, MAX_BATTERY]
    * ``rescued``        — tuple of booleans (one per target)
    * ``charged``        — boolean, True once the charger has been visited
                           (prevents repeated +5 reward exploit)

    A state is terminal when battery = 0 or all targets are rescued.
    Terminal states have V*(s) = 0.

    Attributes:
        grid (np.ndarray):       5 × 5 cell-type grid
        valid_pos (list):        non-blocked (row, col) positions
        states (list):           every valid state tuple
        terminal_states (set):   terminal subset
        state_to_idx (dict):     state → list index
    """

    def __init__(self):
        """Build grid, enumerate positions and states, mark terminals."""
        self.grid = np.array(GRID_MAP, dtype=np.int32)
        self.valid_pos = [
            (r, c)
            for r in range(ROWS) for c in range(COLS)
            if self.grid[r, c] != BLOCKED
        ]
        self.states = self._enumerate_states()
        self.terminal_states = self._find_terminals()
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}

    def _enumerate_states(self):
        """Return every valid MDP state.

        Iterates over valid positions × battery levels ×
        rescue-status combinations × charger-visited flag.

        Returns:
            list[tuple]: all state tuples
        """
        rescue_combos = list(
            itertools.product([False, True], repeat=NUM_TARGETS)
        )
        return [
            (r, c, b, rc, ch)
            for (r, c) in self.valid_pos
            for b in range(MAX_BATTERY + 1)
            for rc in rescue_combos
            for ch in (False, True)
        ]

    def _find_terminals(self):
        """States where battery = 0 or all targets rescued.

        Returns:
            set[tuple]: terminal states
        """
        return {
            s for s in self.states
            if s[2] == 0 or all(s[3])
        }

    def cell_type(self, r, c, rescued):
        """Effective cell type, accounting for already-rescued targets.

        A rescue cell becomes SAFE once that target has been collected.

        Args:
            r (int): row
            c (int): column
            rescued (tuple): rescue-status booleans

        Returns:
            int: cell-type constant
        """
        ct = self.grid[r, c]
        if ct == RESCUE:
            for idx, pos in enumerate(RESCUE_POS):
                if (r, c) == pos and rescued[idx]:
                    return SAFE
        return ct

    @staticmethod
    def _move(r, c, action):
        """Candidate position after applying *action*, clipped to bounds.

        Args:
            r (int): current row
            c (int): current column
            action (int): UP / DOWN / LEFT / RIGHT / HOVER

        Returns:
            tuple[int, int]: candidate (row, col)
        """
        if action == HOVER:
            return r, c
        moves = {
            UP:    (max(r - 1, 0),        c),
            DOWN:  (min(r + 1, ROWS - 1), c),
            LEFT:  (r, max(c - 1, 0)),
            RIGHT: (r, min(c + 1, COLS - 1)),
        }
        return moves[action]

    def transitions(self, state, action):
        """All probabilistic outcomes for (state, action).

        Mirrors DroneRescueEnv.step() exactly: wind stochasticity,
        battery mechanics, blocked-cell bounce, charging, rescue
        collection, and reward assignment.

        The charging-station reward (+5) is granted only on the FIRST
        arrival per episode, preventing an infinite-horizon exploit
        where the optimal policy oscillates at the charger.

        Args:
            state (tuple): (row, col, battery, rescued, charged)
            action (int):  intended action

        Returns:
            list[tuple]: [(probability, next_state, reward), ...]
        """
        r, c, bat, rescued, charged = state

        if state in self.terminal_states:
            return [(1.0, state, 0.0)]

        cur_cell = self.cell_type(r, c, rescued)

        # Wind stochasticity: 20 % chance of random redirect on wind cells
        aprob = {}
        if cur_cell == WIND:
            if action in (UP, DOWN, LEFT, RIGHT):
                for a in (UP, DOWN, LEFT, RIGHT):
                    aprob[a] = (0.8 if a == action else 0.0) + WIND_PROB / 4
            else:
                aprob[HOVER] = 1.0 - WIND_PROB
                for a in (UP, DOWN, LEFT, RIGHT):
                    aprob[a] = WIND_PROB / 4
        else:
            aprob[action] = 1.0

        out = {}
        for act, prob in aprob.items():
            nr, nc = self._move(r, c, act)

            if act == HOVER and cur_cell == CHARGING:
                nb = min(bat + 2, MAX_BATTERY)
            else:
                nb = bat - BATTERY_COST

            if self.grid[nr, nc] == BLOCKED:
                nr, nc = r, c

            dest_cell = self.cell_type(nr, nc, rescued)
            if dest_cell == CHARGING:
                nb = MAX_BATTERY
            nb = max(nb, 0)

            new_rescued = list(rescued)
            if dest_cell == RESCUE:
                for idx, pos in enumerate(RESCUE_POS):
                    if (nr, nc) == pos and not rescued[idx]:
                        new_rescued[idx] = True
            new_rescued = tuple(new_rescued)

            new_charged = charged
            if dest_cell == CHARGING:
                new_charged = True

            first_charger_visit = (dest_cell == CHARGING and not charged)

            if dest_cell == RESCUE:
                reward = R_RESCUE
            elif dest_cell == DANGER:
                reward = R_DANGER
            elif nb == 0:
                reward = R_DEAD
            elif first_charger_visit:
                reward = R_CHARGE
            else:
                reward = R_STEP

            ns = (nr, nc, nb, new_rescued, new_charged)
            if ns in out:
                out[ns][0] += prob
            else:
                out[ns] = [prob, reward]

        return [(p, ns, rw) for ns, (p, rw) in out.items()]


# ================================================================
# Section 3 — Expected Outcome 2: Value Iteration
# ================================================================

def value_iteration(mdp, gamma=GAMMA, theta=THETA):
    """Compute V*(s) and π*(s) via the Bellman optimality backup.

    Sweeps:
        V_{k+1}(s) = max_a  Σ P(s'|s,a) · [R(s,a,s') + γ · V_k(s')]

    until max |V_{k+1}(s) − V_k(s)| < θ.

    Covers assignment requirements:
      ✓ Enumerates reachable states
      ✓ Computes optimal V*(s) and π*(s)
      ✓ Stopping threshold θ = 10⁻³
      ✓ Reports convergence iterations, runtime, final delta

    Args:
        mdp   (DroneRescueMDP): MDP model
        gamma (float):          discount factor
        theta (float):          convergence threshold

    Returns:
        V       (dict):  state → optimal value
        policy  (dict):  state → optimal action
        deltas  (list):  per-iteration max change
    """
    print("\n  Running Value Iteration ...", end="", flush=True)

    V = {s: 0.0 for s in mdp.states}
    deltas = []
    start = time.time()
    iteration = 0

    while True:
        iteration += 1
        delta = 0.0

        for s in mdp.states:
            if s in mdp.terminal_states:
                continue
            v_old = V[s]
            best = float("-inf")
            for a in ACTIONS:
                q = sum(
                    p * (rw + gamma * V[ns])
                    for p, ns, rw in mdp.transitions(s, a)
                )
                if q > best:
                    best = q
            V[s] = best
            delta = max(delta, abs(v_old - best))

        deltas.append(delta)

        if delta < theta:
            break

    elapsed = time.time() - start

    policy = {}
    for s in mdp.states:
        if s in mdp.terminal_states:
            policy[s] = HOVER
            continue
        best_a, best_q = HOVER, float("-inf")
        for a in ACTIONS:
            q = sum(
                p * (rw + gamma * V[ns])
                for p, ns, rw in mdp.transitions(s, a)
            )
            if q > best_q:
                best_q = q
                best_a = a
        policy[s] = best_a

    print(" CONVERGED")
    print(f"    Iterations : {iteration}")
    print(f"    Runtime    : {elapsed:.4f} s")
    print(f"    Final δ    : {deltas[-1]:.10f}")

    return V, policy, deltas


# ================================================================
# Section 4 — Expected Outcome 2: Policy Iteration
# ================================================================

def policy_iteration(mdp, gamma=GAMMA, theta=THETA):
    """Compute V*(s) and π*(s) via policy evaluation + improvement.

    Alternates:
      1. Policy Evaluation — solve V^π until δ < θ
      2. Policy Improvement — greedy update; check stability

    Args:
        mdp   (DroneRescueMDP): MDP model
        gamma (float):          discount factor
        theta (float):          convergence threshold for evaluation

    Returns:
        V       (dict):  state → optimal value
        policy  (dict):  state → optimal action
        history (list):  per-outer-iteration metadata
    """
    print("  Running Policy Iteration ...", end="", flush=True)

    V = {s: 0.0 for s in mdp.states}
    policy = {
        s: (HOVER if s in mdp.terminal_states else DOWN)
        for s in mdp.states
    }

    history = []
    start = time.time()
    outer = 0
    stable = False

    while not stable:
        outer += 1

        # Phase 1: Policy Evaluation
        eval_iter = 0
        eval_deltas = []
        while True:
            eval_iter += 1
            delta = 0.0
            for s in mdp.states:
                if s in mdp.terminal_states:
                    continue
                v_old = V[s]
                a = policy[s]
                V[s] = sum(
                    p * (rw + gamma * V[ns])
                    for p, ns, rw in mdp.transitions(s, a)
                )
                delta = max(delta, abs(v_old - V[s]))
            eval_deltas.append(delta)
            if delta < theta:
                break

        # Phase 2: Policy Improvement
        stable = True
        changes = 0
        for s in mdp.states:
            if s in mdp.terminal_states:
                continue
            old_a = policy[s]
            best_a, best_q = old_a, float("-inf")
            for a in ACTIONS:
                q = sum(
                    p * (rw + gamma * V[ns])
                    for p, ns, rw in mdp.transitions(s, a)
                )
                if q > best_q:
                    best_q = q
                    best_a = a
            policy[s] = best_a
            if best_a != old_a:
                stable = False
                changes += 1

        history.append({
            "outer": outer,
            "eval_iters": eval_iter,
            "eval_deltas": eval_deltas,
            "policy_changes": changes,
        })

    elapsed = time.time() - start
    total_evals = sum(h["eval_iters"] for h in history)

    print(" CONVERGED")
    print(f"    Outer iterations       : {outer}")
    print(f"    Total eval sweeps      : {total_evals}")
    print(f"    Runtime                : {elapsed:.4f} s")

    return V, policy, history


# ================================================================
# Section 5 — Expected Outcome 3: Policy Visualisation & Simulation
# ================================================================
# Text-based value grids, policy grids with directional arrows,
# and deterministic trajectory simulation.
# ================================================================

def grid_values(V, bat=MAX_BATTERY, rescued=(False, False), charged=False):
    """Extract V*(s) into a 5×5 grid for a fixed state slice.

    Blocked cells are NaN for distinct heatmap rendering.

    Args:
        V (dict): full value function
        bat (int): battery level
        rescued (tuple): rescue-status slice
        charged (bool): charger-visited flag

    Returns:
        np.ndarray: shape (5, 5)
    """
    gv = np.full((ROWS, COLS), np.nan, dtype=np.float64)
    for r in range(ROWS):
        for c in range(COLS):
            s = (r, c, bat, rescued, charged)
            if s in V:
                gv[r, c] = V[s]
    return gv


def grid_policy(policy, bat=MAX_BATTERY, rescued=(False, False),
                charged=False):
    """Extract policy into a {(row,col): action} dict for one slice.

    Args:
        policy (dict): full policy
        bat (int): battery level
        rescued (tuple): rescue-status slice
        charged (bool): charger-visited flag

    Returns:
        dict: (row, col) → action
    """
    gp = {}
    for r in range(ROWS):
        for c in range(COLS):
            s = (r, c, bat, rescued, charged)
            if s in policy:
                gp[(r, c)] = policy[s]
    return gp


def show_grid():
    """Print the environment grid with symbolic labels."""
    print("\n  Environment Grid:")
    for r in range(ROWS):
        print("    " + "  ".join(
            CELL_SYMBOL[GRID_MAP[r][c]] for c in range(COLS)
        ))


def show_values(V, bat=MAX_BATTERY, rescued=(False, False), charged=False):
    """Print V*(s) as a simple text grid for a fixed slice."""
    tag = f"bat={bat}, rescued={rescued}, charged={charged}"
    print(f"\n  V*(s) at [{tag}]")
    for r in range(ROWS):
        vals = []
        for c in range(COLS):
            s = (r, c, bat, rescued, charged)
            if s in V:
                vals.append(f"{V[s]:>8.2f}")
            else:
                vals.append(f"{'---':>8}")
        print("    " + "  ".join(vals))


def show_policy(policy, bat=MAX_BATTERY, rescued=(False, False),
                charged=False):
    """Print the policy as a simple text grid of directional arrows."""
    tag = f"bat={bat}, rescued={rescued}, charged={charged}"
    print(f"\n  Policy at [{tag}]")
    for r in range(ROWS):
        cells = []
        for c in range(COLS):
            s = (r, c, bat, rescued, charged)
            if s in policy:
                cells.append(ACTION_ARROW[policy[s]])
            else:
                cells.append("X")
        print("    " + "  ".join(cells))


def simulate_policy(mdp, policy, max_steps=50):
    """Run the optimal policy from the start state (deterministic).

    Wind zones are resolved deterministically (intended direction
    always succeeds) to show the most-likely trajectory.

    Args:
        mdp       (DroneRescueMDP): MDP model
        policy    (dict):           state → action mapping
        max_steps (int):            episode length cap

    Returns:
        list[tuple]: trajectory as (state, action, reward) triples
    """
    print("\n  Policy Simulation (deterministic, no wind)")

    state = (0, 0, MAX_BATTERY, (False,) * NUM_TARGETS, False)
    traj = []
    total_r = 0.0

    for step in range(max_steps):
        r, c, bat, rescued, charged = state
        if state in mdp.terminal_states:
            reason = ("All rescued!" if all(rescued)
                      else "Battery depleted!")
            print(f"  Step {step}: episode ended - {reason}")
            break

        action = policy.get(state, HOVER)

        nr, nc = mdp._move(r, c, action)
        cur_cell = mdp.cell_type(r, c, rescued)

        if action == HOVER and cur_cell == CHARGING:
            nb = min(bat + 2, MAX_BATTERY)
        else:
            nb = bat - BATTERY_COST

        if mdp.grid[nr, nc] == BLOCKED:
            nr, nc = r, c

        dest = mdp.cell_type(nr, nc, rescued)
        if dest == CHARGING:
            nb = MAX_BATTERY
        nb = max(nb, 0)

        new_rescued = list(rescued)
        if dest == RESCUE:
            for idx, pos in enumerate(RESCUE_POS):
                if (nr, nc) == pos and not rescued[idx]:
                    new_rescued[idx] = True
        new_rescued = tuple(new_rescued)

        new_charged = charged
        if dest == CHARGING:
            new_charged = True

        first_visit = (dest == CHARGING and not charged)

        if dest == RESCUE:
            reward = R_RESCUE
        elif dest == DANGER:
            reward = R_DANGER
        elif nb == 0:
            reward = R_DEAD
        elif first_visit:
            reward = R_CHARGE
        else:
            reward = R_STEP

        total_r += reward

        rescued_count = sum(new_rescued)
        print(f"  Step {step}: ({r},{c})->({nr},{nc}) "
              f"bat={bat}->{nb} action={ACTION_NAME[action]} "
              f"reward={reward:+.0f} total={total_r:+.1f} "
              f"rescued={rescued_count}/{NUM_TARGETS}")

        traj.append((state, action, reward))
        state = (nr, nc, nb, new_rescued, new_charged)

    print(f"  Cumulative reward: {total_r:+.1f}")
    return traj


def plot_optimal_path(mdp, policy, V, save_dir="."):
    """Plot the optimal traversal path on the grid with V*(s) background.

    Shows the drone's step-by-step route from start to rescuing both
    targets, overlaid on the value heatmap at full battery.

    Args:
        mdp    (DroneRescueMDP): MDP model
        policy (dict):           state -> action
        V      (dict):           state -> value
        save_dir (str):          output directory
    """
    grid_arr = np.array(GRID_MAP, dtype=np.int32)
    blocked_mask = np.array(
        [[grid_arr[r, c] == BLOCKED for c in range(COLS)]
         for r in range(ROWS)]
    )

    gv = grid_values(V, bat=MAX_BATTERY, rescued=(False, False), charged=False)
    masked = np.ma.array(gv, mask=blocked_mask)
    valid_vals = gv[~blocked_mask]
    vmin = np.nanmin(valid_vals) if valid_vals.size else 0
    vmax = np.nanmax(valid_vals) if valid_vals.size else 1

    state = (0, 0, MAX_BATTERY, (False,) * NUM_TARGETS, False)
    positions = [(0, 0)]
    for _ in range(50):
        r, c, bat, rescued, charged = state
        if state in mdp.terminal_states:
            break
        action = policy.get(state, HOVER)
        nr, nc = mdp._move(r, c, action)
        if mdp.grid[nr, nc] == BLOCKED:
            nr, nc = r, c

        nb = bat - BATTERY_COST
        dest = mdp.cell_type(nr, nc, rescued)
        if dest == CHARGING:
            nb = MAX_BATTERY
        nb = max(nb, 0)

        new_rescued = list(rescued)
        if dest == RESCUE:
            for idx, pos in enumerate(RESCUE_POS):
                if (nr, nc) == pos and not rescued[idx]:
                    new_rescued[idx] = True
        new_rescued = tuple(new_rescued)

        new_charged = charged or (dest == CHARGING)
        positions.append((nr, nc))
        state = (nr, nc, nb, tuple(new_rescued), new_charged)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(masked, cmap="RdYlGn", interpolation="nearest",
              vmin=vmin, vmax=vmax)

    for i in range(ROWS):
        for j in range(COLS):
            if blocked_mask[i, j]:
                ax.text(j, i, "X", ha="center", va="center",
                        fontsize=11, color="white", fontweight="bold")
            else:
                label = CELL_SYMBOL.get(grid_arr[i, j], "")
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=9, color="black", alpha=0.6)

    cols_path = [p[1] for p in positions]
    rows_path = [p[0] for p in positions]
    ax.plot(cols_path, rows_path, "b-o", ms=6, lw=2, alpha=0.8,
            markerfacecolor="white", markeredgewidth=1.5)

    ax.plot(cols_path[0], rows_path[0], "s", ms=12,
            color="blue", label="Start")
    ax.plot(cols_path[-1], rows_path[-1], "*", ms=16,
            color="red", label="End")

    for step_i, (pr, pc) in enumerate(positions):
        ax.annotate(str(step_i), (pc, pr), fontsize=7,
                    ha="center", va="bottom", color="blue",
                    xytext=(0, 6), textcoords="offset points")

    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(f"Optimal Policy Traversal Path (Team {G})\n"
                 f"Start (0,0) -> Rescue T0 (0,4) -> Charger (2,2) -> Rescue T1 (4,0)")
    ax.legend(loc="upper right")

    plt.tight_layout()
    path = os.path.join(save_dir, "optimal_path.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Optimal path plot saved -> {path}")
    return path


# ================================================================
# Section 6 — Expected Outcome 2: Convergence Plotting
# ================================================================

def plot_convergence(vi_deltas, pi_history, path="convergence_plot.png"):
    """Save a two-panel convergence plot (VI vs PI).

    Left  — VI delta per sweep.
    Right — PI evaluation deltas with policy-improvement markers.

    Args:
        vi_deltas  (list[float]): from value_iteration()
        pi_history (list[dict]):  from policy_iteration()
        path       (str):         output PNG path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Convergence Analysis — Value Iteration vs Policy Iteration",
        fontsize=13, fontweight="bold",
    )

    xs = range(1, len(vi_deltas) + 1)
    ax1.plot(xs, vi_deltas, "b-o", ms=2, lw=1.0)
    ax1.axhline(THETA, color="r", ls="--", lw=0.8, label=f"θ = {THETA}")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("δ  (max |ΔV|)")
    ax1.set_title("Value Iteration")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    all_d, bounds = [], [0]
    for h in pi_history:
        all_d.extend(h["eval_deltas"])
        bounds.append(len(all_d))
    xs2 = range(1, len(all_d) + 1)
    ax2.plot(xs2, all_d, "g-o", ms=2, lw=1.0)
    ax2.axhline(THETA, color="r", ls="--", lw=0.8, label=f"θ = {THETA}")
    for i, b in enumerate(bounds[1:-1], 1):
        ax2.axvline(b, color="orange", ls=":", alpha=0.6)
        ax2.text(b, max(all_d) * 0.4, f"PI-{i}", fontsize=7,
                 ha="center", color="orange")
    ax2.set_xlabel("Cumulative Eval Sweep")
    ax2.set_ylabel("δ  (max |ΔV|)")
    ax2.set_title("Policy Iteration")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Convergence plot saved → {path}")
    plt.close()


# ================================================================
# Section 7 — Expected Outcome 2: Comparative Analysis
# ================================================================

def compare(vi_res, pi_res):
    """Print a side-by-side comparison of VI and PI results.

    Checks whether V* and π* agree and compares iteration counts.

    Args:
        vi_res (tuple): (V, policy, deltas)  from value_iteration
        pi_res (tuple): (V, policy, history) from policy_iteration
    """
    V_vi, p_vi, d_vi = vi_res
    V_pi, p_pi, h_pi = pi_res

    pi_evals = sum(h["eval_iters"] for h in h_pi)

    print("\n  Comparative Analysis (VI vs PI):")

    max_diff = max(abs(V_vi[s] - V_pi[s]) for s in V_vi if s in V_pi)
    agree = sum(1 for s in p_vi if s in p_pi and p_vi[s] == p_pi[s])
    total = len(p_vi)

    print(f"  VI sweeps: {len(d_vi)}, PI total eval sweeps: {pi_evals} "
          f"({len(h_pi)} outer iterations)")
    print(f"  Max |V*_VI - V*_PI| = {max_diff:.10f}")
    print(f"  Policy agreement: {agree}/{total} ({100 * agree / total:.1f}%)")

    if max_diff < THETA:
        print("  -> Both converge to the same optimal value function (within theta).")
    if agree == total:
        print("  -> Both yield identical optimal policies.")


# ================================================================
# Section 8 — Expected Outcome 4: State-Value Heatmap Analysis
#              (PARTIAL — covers heatmap plots & pattern analysis)
# ================================================================

def plot_state_value_heatmaps(V, save_dir="."):
    """Generate a 2×3 panel of V*(s) heatmaps for meaningful slices.

    Each panel fixes battery level and rescue status, varying only
    drone position.  Blocked cells are rendered distinctly.

    Args:
        V (dict):       full value function from DP
        save_dir (str): directory for the output PNG

    Returns:
        str: path to the saved heatmap image
    """
    slices = [
        {"bat": MAX_BATTERY, "rescued": (False, False), "charged": False,
         "title": f"Battery={MAX_BATTERY}, No rescues"},
        {"bat": MAX_BATTERY // 2, "rescued": (False, False), "charged": False,
         "title": f"Battery={MAX_BATTERY // 2}, No rescues"},
        {"bat": 3, "rescued": (False, False), "charged": False,
         "title": "Battery=3, No rescues"},
        {"bat": 1, "rescued": (False, False), "charged": False,
         "title": "Battery=1, No rescues"},
        {"bat": MAX_BATTERY, "rescued": (True, False), "charged": False,
         "title": f"Battery={MAX_BATTERY}, Target 0 rescued"},
        {"bat": MAX_BATTERY, "rescued": (False, True), "charged": False,
         "title": f"Battery={MAX_BATTERY}, Target 1 rescued"},
    ]

    grid_arr = np.array(GRID_MAP, dtype=np.int32)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        "State-Value V*(s) Heatmaps — Drone Position vs Battery & Rescue Status\n"
        f"(Grid: {ROWS}×{COLS},  γ = {GAMMA},  Team {G})",
        fontsize=14, fontweight="bold",
    )

    for ax, sl in zip(axes.flat, slices):
        gv = grid_values(V, bat=sl["bat"], rescued=sl["rescued"],
                         charged=sl["charged"])

        blocked_mask = np.array(
            [[grid_arr[r, c] == BLOCKED for c in range(COLS)]
             for r in range(ROWS)]
        )
        masked = np.ma.array(gv, mask=blocked_mask)

        valid_vals = gv[~blocked_mask]
        vmin = np.nanmin(valid_vals) if valid_vals.size else 0
        vmax = np.nanmax(valid_vals) if valid_vals.size else 1

        im = ax.imshow(masked, cmap="RdYlGn", interpolation="nearest",
                       vmin=vmin, vmax=vmax)
        ax.set_title(sl["title"], fontsize=10, fontweight="bold")

        for i in range(ROWS):
            for j in range(COLS):
                if blocked_mask[i, j]:
                    ax.text(j, i, "X", ha="center", va="center",
                            fontsize=12, color="white", fontweight="bold")
                else:
                    val = gv[i, j]
                    label = CELL_SYMBOL.get(grid_arr[i, j], "")
                    mid = (vmin + vmax) / 2
                    color = "black" if val > mid else "white"
                    ax.text(j, i, f"{val:.1f}\n{label}", ha="center",
                            va="center", fontsize=8, color=color)

        ax.set_xticks(range(COLS))
        ax.set_yticks(range(ROWS))
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    path = os.path.join(save_dir, "task5_state_value_heatmaps.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  State-value heatmaps saved → {path}")
    return path


def print_state_value_analysis():
    """Print analysis of observed V*(s) patterns (Expected Outcome 4)."""
    print("\n  State-Value Analysis:")
    print("  - Cells near rescue targets (0,4) and (4,0) have highest V*(s)")
    print("  - V*(s) drops sharply with lower battery; at bat=1, most cells are negative")
    print("  - Danger zones (1,2), (3,0), (4,3) reduce nearby V*(s) by -10 penalty")
    print("  - Charger at (2,2) boosts V*(s) via battery refill + one-time +5 reward")
    print("  - After rescuing target 0, value landscape shifts toward target 1 at (4,0)")


# ================================================================
# Section 9 — Expected Outcome 5: DP Scalability Discussion
#              (PARTIAL — textual analysis)
# ================================================================

def print_scalability_discussion(mdp):
    """Print the DP scalability discussion (Expected Outcome 5).

    Covers the curse of dimensionality, scaling projections, and
    the case for Deep RL as an alternative.

    Args:
        mdp (DroneRescueMDP): used for concrete state-space counts
    """
    n_states = len(mdp.states)
    n_pos = len(mdp.valid_pos)
    n_term = len(mdp.terminal_states)

    print(f"\n  DP Scalability Discussion:")
    print(f"  Current: {n_pos} positions x {MAX_BATTERY+1} battery x "
          f"{2**NUM_TARGETS} rescue combos x 2 charger = {n_states} states "
          f"({n_term} terminal)")
    print(f"  This is tractable for tabular DP.")
    print()
    print(f"  Scaling estimates:")
    print(f"  - 10x10 grid, 2 targets: ~11,520 states (4x current, still tractable)")
    print(f"  - 10x10 grid, 5 targets: ~92,160 states (2^5=32 rescue combos)")
    print(f"  - Add 3 weather modes: ~276,480 states (memory becomes an issue)")
    print(f"  - Add 10 weather modes: ~921,600 states (minutes per sweep)")
    print()
    print(f"  Why DP becomes impractical:")
    print(f"  - State space grows exponentially with each dimension")
    print(f"  - Transition tables become too large for memory")
    print(f"  - Continuous states (GPS, analog battery) need infinite tables")
    print(f"  - Each sweep iterates over ALL states with no shortcuts")
    print()
    print(f"  Deep RL alternatives:")
    print(f"  - DQN: function approximation replaces V/Q tables")
    print(f"  - Policy Gradient (PPO, SAC): learn parameterized policies")
    print(f"  - Model-based RL (Dreamer, MuZero): learn world model + plan")
    print(f"  - Trade-off: no convergence guarantees, needs hyperparameter tuning")
    print()
    print(f"  Real-world drones need continuous states, partial observability,")
    print(f"  multi-agent coordination - tabular DP is infeasible for production.")


# ================================================================
# Section 10 — Simulation Infrastructure (Logging & Action Modes)
# ================================================================
# Structured logging, timestamped directories, Pygame rendering,
# video export, and multiple action modes (algorithm / custom /
# random).  Requires gymnasium + pygame + imageio.
# ================================================================

LOG_ROOT = Path("logs")
LOG_SEPARATOR = "*" * 88

ALGORITHM_MODE = "algorithm"
CUSTOM_MODE    = "custom"
RANDOM_MODE    = "random"
VALID_MODES    = {ALGORITHM_MODE, CUSTOM_MODE, RANDOM_MODE}

CUSTOM_ACTIONS = [3, 3, 3, 3, 2, 1, 1, 2, 2, 1, 1, 2]


def create_log_directory():
    """Create a timestamped log directory under ``logs/``.

    Returns:
        Path: Path to the created directory.
    """
    folder = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = LOG_ROOT / folder
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def setup_logger(log_dir):
    """Configure dual logging (file + console).

    Args:
        log_dir (Path): Directory for the log file.

    Returns:
        logging.Logger: Configured logger.
    """
    log_file = log_dir / "simulation.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s]: %(message)s",
        force=True,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def initialize_environment(log_dir, render_mode="human"):
    """Create and reset the DroneRescueEnv.

    Args:
        log_dir (Path): Directory for renderer output.
        render_mode (str or None): ``"human"`` for Pygame; ``None`` headless.

    Returns:
        tuple: (env, observation, info)
    """
    env = DroneRescueEnv(render_mode=render_mode, log_dir=str(log_dir))
    obs, info = env.reset(seed=42)
    env.render()
    return env, obs, info


def dp_policy_to_env_actions(env, policy_full, max_steps=MAX_STEPS):
    """Convert the full DP policy into a sequence of env-level actions.

    The DP policy maps ``(row, col, battery, rescued, charged)`` to
    actions.  This function simulates the MDP transitions locally
    (deterministic, ignoring wind) to produce an action list that
    ``simulation_loop`` can execute step-by-step.

    Args:
        env (DroneRescueEnv): Environment instance (for grid info).
        policy_full (dict): state -> action from Value/Policy Iteration.
        max_steps (int): Maximum actions to generate.

    Returns:
        list[int]: Action sequence.
    """
    mdp_local = DroneRescueMDP()
    state = (0, 0, MAX_BATTERY, (False,) * NUM_TARGETS, False)
    actions = []

    for _ in range(max_steps):
        if state in mdp_local.terminal_states:
            break
        action = policy_full.get(state, HOVER)
        actions.append(action)

        outcomes = mdp_local.transitions(state, action)
        state = max(outcomes, key=lambda t: t[0])[1]

    return actions


def get_action_sequence(env, mode, policy_full=None):
    """Return an action sequence for the chosen mode.

    Args:
        env (DroneRescueEnv): Environment instance.
        mode (str): One of ``algorithm``, ``custom``, ``random``.
        policy_full (dict or None): DP policy (needed for algorithm mode).

    Returns:
        list[int]: Action sequence.

    Raises:
        ValueError: If mode is invalid.
    """
    if mode not in VALID_MODES:
        raise ValueError(
            f"Invalid mode {mode!r}. Valid: {', '.join(sorted(VALID_MODES))}"
        )

    if mode == CUSTOM_MODE:
        return list(CUSTOM_ACTIONS)

    if mode == ALGORITHM_MODE:
        if policy_full is None:
            return [env.action_space.sample() for _ in range(MAX_STEPS)]
        actions = dp_policy_to_env_actions(env, policy_full)
        if not actions:
            return [env.action_space.sample() for _ in range(MAX_STEPS)]
        return actions

    return [env.action_space.sample() for _ in range(MAX_STEPS * 3)]


def log_step(logger, info, reward, total_reward):
    """Log a single simulation step.

    Args:
        logger: Logger instance.
        info (dict): Environment info dict.
        reward (float): Step reward.
        total_reward (float): Cumulative reward.
    """
    policy_type = (
        "Algorithm" if info.get("using_algorithm_policy", False)
        else "Fallback"
    )
    logger.info(
        "Step: %s | Position: %s | Battery: %s | Reward: %s "
        "| Action: %s | Policy: %s | Total Reward: %s",
        info["step_count"], info["state"], info["battery_level"],
        reward, info["action_taken"], policy_type, total_reward,
    )


def log_rescue_progress(logger, info):
    """Log rescue target progress.

    Args:
        logger: Logger instance.
        info (dict): Environment info dict.
    """
    rescue = info.get("rescue_target_state", {})
    rescued_count = sum(1 for v in rescue.values() if not v)
    logger.info("Rescue Targets Rescued: %s", rescued_count)
    logger.info("Rescue Status: %s", rescue)


def handle_termination(logger, terminated, truncated, total_reward, info):
    """Log episode end and return whether to stop.

    Args:
        logger: Logger instance.
        terminated (bool): True if episode terminated.
        truncated (bool): True if episode was truncated.
        total_reward (float): Cumulative reward.
        info (dict): Environment info dict.

    Returns:
        bool: True if simulation should stop.
    """
    rescue = info.get("rescue_target_state", {})
    battery = info.get("battery_level", -1)
    rescued_count = sum(1 for v in rescue.values() if not v)

    logger.info(
        "Rescue Targets Rescued: %s | Rescue Status: %s",
        rescued_count, rescue,
    )
    logger.info(LOG_SEPARATOR)

    if terminated:
        if rescued_count == len(rescue):
            reason = "All Targets Rescued"
        elif battery == 0:
            reason = "Battery Depleted"
        else:
            reason = "Environment/Agent Failure"
        logger.info(
            "Episode Terminated | Final Reward: %s | Reason: %s",
            total_reward, reason,
        )
        logger.info(LOG_SEPARATOR)
        return True

    if truncated:
        logger.warning(
            "Episode Truncated due to max steps | Final Reward: %s",
            total_reward,
        )
        logger.info(LOG_SEPARATOR)
        return True

    return False


def simulation_loop(env, logger, mode, policy_full=None):
    """Run the simulation loop until termination.

    Args:
        env (DroneRescueEnv): Environment instance.
        logger: Logger instance.
        mode (str): Action selection mode.
        policy_full (dict or None): DP policy for algorithm mode.
    """
    total_reward = 0.0
    actions = get_action_sequence(env, mode, policy_full)
    logger.info("Action Mode Selected: %s", mode)

    if not actions:
        logger.warning("No actions available to execute.")
        return

    try:
        for action in actions:
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            logger.info("=" * 60)
            log_step(logger, info, reward, total_reward)
            log_rescue_progress(logger, info)

            env.render()
            if handle_termination(
                logger, terminated, truncated, total_reward, info
            ):
                break
    except Exception:
        logger.exception("Simulation failed while executing an action.")
        raise


def cleanup_environment(env, logger):
    """Close renderer, save video, and flush logs.

    Args:
        env (DroneRescueEnv): Environment instance.
        logger: Logger instance.
    """
    logger.info("------------- END SIMULATION -------------")
    try:
        env.close()
    except Exception:
        logger.exception("Failed to close environment cleanly.")
        raise
    finally:
        logging.shutdown()


# ================================================================
# Section 11 — Simulation Runner Entry Point
# ================================================================

def run_simulation(mode=ALGORITHM_MODE, policy_full=None,
                   render_mode="human"):
    """Run the full drone rescue simulation with logging and rendering.

    This wires together:
      - Timestamped log directory creation
      - Dual logging (file + console)
      - DroneRescueEnv initialisation with optional Pygame rendering
      - DP-policy-driven or custom/random action execution
      - Video export (simulation.mp4) and structured log output

    Args:
        mode (str): ``"algorithm"``, ``"custom"``, or ``"random"``.
        policy_full (dict or None): DP policy for algorithm mode.
        render_mode (str or None): ``"human"`` for GUI, ``None`` headless.
    """
    if not _HAS_GYMNASIUM:
        print("\n  [SKIP] Simulation requires gymnasium + pygame + imageio.")
        print("         Install them to enable live rendering & video export.")
        return

    log_dir = create_log_directory()
    logger = setup_logger(log_dir)
    env = None

    logger.info("------------- START SIMULATION -------------")
    try:
        env, obs, info = initialize_environment(log_dir, render_mode)
        logger.info(
            "Initial Position: %s | Initial Battery: %s | Observation: %s",
            info["state"], info["battery_level"], obs,
        )
        simulation_loop(env, logger, mode, policy_full)
    except Exception:
        logger.exception("Simulation stopped because of an unexpected error.")
        raise
    finally:
        if env is not None:
            cleanup_environment(env, logger)

    print(f"\n  Simulation artifacts saved → {log_dir}/")
    print(f"    simulation.log  — structured step-by-step log")
    print(f"    simulation.mp4  — rendered video (if Pygame was enabled)")


# ================================================================
# Section 12 — Main Execution Block
# ================================================================

if __name__ == "__main__":

    # ---- Header ----
    ts, vm = get_vm_info()
    print(f"Team: {G}")
    print(f"Timestamp: {ts}")
    # print(f"VM ID: {vm}")

    # ---- Environment config ----
    print(f"\nEnvironment: {ROWS}x{COLS} grid, battery={MAX_BATTERY}, "
          f"wind={WIND_PROB*100:.0f}%, gamma={GAMMA}, theta={THETA}")
    print(f"Rescue targets: {RESCUE_POS}")
    print(f"Charging: [(2,2)], Danger: [(1,2),(3,0),(4,3)], "
          f"Blocked: [(1,1),(3,2)], Wind: [(0,2),(1,4)]")
    show_grid()

    # ---- State space ----
    mdp = DroneRescueMDP()
    print(f"\nState space: {len(mdp.states)} total "
          f"({len(mdp.terminal_states)} terminal, "
          f"{len(mdp.states) - len(mdp.terminal_states)} non-terminal)")
    print(f"State = (row, col, battery, rescued_tuple, charger_visited)")
    print(f"Start = (0, 0, {MAX_BATTERY}, {(False,)*NUM_TARGETS}, False)")

    # ---- Value Iteration ----
    vi_V, vi_pi, vi_deltas = value_iteration(mdp)
    show_values(vi_V)
    show_policy(vi_pi)

    # ---- Policy Iteration ----
    pi_V, pi_pi, pi_hist = policy_iteration(mdp)

    # ---- Comparative Analysis ----
    compare(
        (vi_V, vi_pi, vi_deltas),
        (pi_V, pi_pi, pi_hist),
    )

    # ---- Policy Simulation ----
    simulate_policy(mdp, vi_pi)

    # ---- Optimal Path Plot ----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_optimal_path(mdp, vi_pi, vi_V, save_dir=script_dir)

    # ---- V* after partial rescue (policy shifts toward remaining target) ----
    print("\n  After rescuing target 0:")
    show_values(vi_V, bat=MAX_BATTERY, rescued=(True, False))
    show_policy(vi_pi, bat=MAX_BATTERY, rescued=(True, False))

    # ---- Convergence Plot ----
    conv_path = os.path.join(script_dir, "convergence_plot.png")
    plot_convergence(vi_deltas, pi_hist, path=conv_path)

    # ---- State-Value Heatmaps ----
    plot_state_value_heatmaps(vi_V, save_dir=script_dir)
    print_state_value_analysis()

    # ---- Scalability Discussion ----
    print_scalability_discussion(mdp)

    # ---- Summary ----
    s0 = (0, 0, MAX_BATTERY, (False, False), False)
    print(f"\nSummary:")
    print(f"  V*(start) = {vi_V[s0]:.4f}, policy(start) = {ACTION_NAME[vi_pi[s0]]}")
    print(f"  VI: {len(vi_deltas)} iterations, "
          f"PI: {len(pi_hist)} outer ({sum(h['eval_iters'] for h in pi_hist)} eval sweeps)")

    # ---- Live Simulation with Pygame + Logging ----
    # Pass --simulate to run the interactive simulation after DP analysis.
    # Modes: --simulate algorithm  (default, uses DP policy)
    #        --simulate custom     (uses CUSTOM_ACTIONS list)
    #        --simulate random     (random actions)
    # Add --headless to skip Pygame window (still saves log).
    if "--simulate" in sys.argv:
        idx = sys.argv.index("--simulate")
        sim_mode = (
            sys.argv[idx + 1]
            if idx + 1 < len(sys.argv) and sys.argv[idx + 1] in VALID_MODES
            else ALGORITHM_MODE
        )
        render = None if "--headless" in sys.argv else "human"
        print(f"\n  Starting live simulation (mode={sim_mode}, "
              f"render={render or 'headless'})...")
        run_simulation(
            mode=sim_mode,
            policy_full=vi_pi,
            render_mode=render,
        )
    else:
        print("\n  Tip: Add --simulate to run Pygame simulation with logging.")
        print("       python3 \"TEAM_181 - DP.py\" --simulate algorithm")
        print("       python3 \"TEAM_181 - DP.py\" --simulate custom")
        print("       python3 \"TEAM_181 - DP.py\" --simulate random")
        print("       Add --headless to skip the GUI window.")

    print("\n  Done.")
