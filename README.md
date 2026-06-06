# TEAM 181 — Dynamic Programming (Part 2)

**Course:** Deep Reinforcement Learning — Lab Assignment 1  
**Part:** 2 — Autonomous Drone Rescue Using Dynamic Programming  
**Team:** 181

---

## How to Run

### DP Analysis Only

```bash
cd team-181
pip install -r requirements.txt
python3 "TEAM_181 - DP.py"
```

Only **numpy** and **matplotlib** are needed (listed in `requirements.txt`).

### DP + Live Simulation (Pygame rendering, logging, video export)

```bash
# Install extra dependencies
pip install gymnasium pygame imageio

# Run with simulation (uses DP-computed optimal policy)
python3 "TEAM_181 - DP.py" --simulate algorithm

# Or use custom / random action modes
python3 "TEAM_181 - DP.py" --simulate custom
python3 "TEAM_181 - DP.py" --simulate random

# Headless (log file only, no GUI window)
python3 "TEAM_181 - DP.py" --simulate algorithm --headless
```

Simulation artifacts are saved under `logs/<timestamp>/`:
- `simulation.log` — structured step-by-step log
- `simulation.mp4` — rendered video (when Pygame is enabled)

---

## Task Completion Status

| Task | Description                                                          | Status   |
|------|----------------------------------------------------------------------|----------|
| 3    | Drone rescue environment design + policy visualization               | COMPLETE |
| 4    | DP solution (value/policy iteration + convergence logic)             | COMPLETE |
| 5    | State value analysis + DP scalability discussion + final integration | PARTIAL  |

### Expected Outcomes Breakdown

| # | Expected Outcome                    | Marks | Status   | Covered By         |
|---|-------------------------------------|-------|----------|--------------------|
| 1 | Custom Drone Rescue Environment     | 1     | COMPLETE | Task 3             |
| 2 | Dynamic Programming Solution        | 2     | COMPLETE | Task 4             |
| 3 | Policy Visualisation                | 1     | COMPLETE | Task 3 + 4         |
| 4 | State-Value Analysis                | 1     | PARTIAL  | Task 5 (heatmaps + textual analysis done) |
| 5 | DP Scalability Discussion           | 1     | PARTIAL  | Task 5 (textual discussion done)          |

---

## File Structure

```
team-181/
├── TEAM_181 - DP.py              ← Main file (run this)
├── gymnasium_env.py              ← Gymnasium DroneRescueEnv (separated)
├── requirements.txt              ← Dependencies
├── README.md                     ← This file
├── convergence_plot.png          ← Generated: VI vs PI convergence
├── task5_state_value_heatmaps.png ← Generated: V*(s) heatmaps
└── optimal_path.png              ← Generated: drone traversal path
```

### Main File Sections (`TEAM_181 - DP.py`)

| Section | Contents                                       |
|---------|-------------------------------------------------|
| 1       | Imports, VM Info, Shared Constants               |
| 2       | MDP Model — state enumeration & transitions      |
| 3       | Value Iteration                                  |
| 4       | Policy Iteration                                 |
| 5       | Policy Visualisation & Simulation (text-based)   |
| 6       | Convergence Plotting (dual-panel PNG)            |
| 7       | Comparative Analysis (VI vs PI)                  |
| 8       | State-Value Heatmap Analysis (Expected Outcome 4)|
| 9       | DP Scalability Discussion (Expected Outcome 5)   |
| 10      | Simulation Infrastructure (logging, action modes)|
| 11      | Simulation Runner Entry Point                    |
| 12      | Main Execution Block                             |

### Gymnasium Environment (`gymnasium_env.py`)

Separated from the main file to keep concerns clean. Contains the
`DroneRescueEnv` class — a Gymnasium-compatible environment with
`reset()`, `step()`, and `render()`. Required only for interactive
simulation (via `Drone_Rescue_DP/run_simulation.py`), **not** for
the DP analysis.

### External Support Files (in `Drone_Rescue_DP/`)

| File               | Purpose                                        |
|--------------------|------------------------------------------------|
| `renderer.py`      | Pygame-based grid visualisation (arrows, heatmap, drone path, video export) |
| `run_simulation.py`| Interactive simulation runner with logging      |

These are optional — the main DP file runs fully standalone.

### Simulation Output (`logs/`)

When `--simulate` is used, each run creates a timestamped directory:

```
logs/2026-06-06_13-05-01/
├── simulation.log   ← Structured step-by-step log
└── simulation.mp4   ← Rendered video (Pygame)
```

### Action Modes

| Mode        | Description                                    |
|-------------|------------------------------------------------|
| `algorithm` | Uses DP-computed optimal policy (VI result)    |
| `custom`    | Uses the CUSTOM_ACTIONS list in the source     |
| `random`    | Samples random actions from the action space   |

---

## Environment Configuration (Group 181, last digit = 1)

| Parameter       | Value                              |
|-----------------|------------------------------------|
| Grid            | 5 × 5                             |
| Battery         | 15 units (odd last digit)          |
| Wind            | 20% stochastic deviation           |
| Targets         | 2 rescue targets: (0,4), (4,0)     |
| Charging        | 1 station: (2,2)                   |
| Danger zones    | 3: (1,2), (3,0), (4,3)            |
| Blocked cells   | 2: (1,1), (3,2)                   |
| Wind zones      | 2: (0,2), (1,4)                   |

### DP Parameters

| Parameter | Value | Notes                           |
|-----------|-------|---------------------------------|
| γ         | 0.99  | Discount factor                 |
| θ         | 10⁻³  | Convergence threshold (per PDF) |

### Rewards

| Event             | Reward |
|-------------------|--------|
| Rescue target     | +20    |
| Charging station  | +5 (first visit only) |
| Step penalty      | −1     |
| Danger zone       | −10    |
| Battery dead      | −20    |

---

## State Representation

```
(row, col, battery, rescued_tuple, charger_visited)
```

- **row, col** — drone position on the 5×5 grid
- **battery** — integer 0 to 15
- **rescued_tuple** — (bool, bool) for each target
- **charger_visited** — bool, prevents repeated +5 exploit

**Total states:** 23 positions × 16 battery levels × 4 rescue combos × 2 charger flags = **2,944**

---

## Generated Output

When the main file runs, it produces:

1. **Console output** — environment config, state-space stats, convergence results, policy simulation, comparative analysis, and summary.
2. **convergence_plot.png** — how fast VI and PI converge.
3. **task5_state_value_heatmaps.png** — V*(s) across battery levels and rescue progress.
4. **optimal_path.png** — the drone's actual route on the grid.

---

## Plot Explanations

### 1. Convergence Plot (`convergence_plot.png`)

This plot shows how quickly each algorithm finds the optimal solution.

**Left panel — Value Iteration:**
- The y-axis (δ) measures how much the value function changed in each sweep. Large δ means the values are still changing; small δ means they have settled.
- For the first ~10 iterations, δ stays high because value information is still spreading across the grid from the reward cells back to distant cells.
- Between iterations 10–14, δ drops sharply below the threshold θ = 0.001 (red dashed line), meaning the algorithm has converged.
- VI converges in 14 sweeps total.

**Right panel — Policy Iteration:**
- Each orange vertical line marks a policy improvement step (there are 10 in total).
- Between improvements, the algorithm runs many evaluation sweeps to fully compute V(s) for the current policy.
- The sawtooth pattern: after each policy change, δ spikes (values need recalculating), then drops as evaluation finishes.
- Spikes get smaller over time because later policy changes are minor.
- PI uses 755 total evaluation sweeps across 10 outer iterations.

**Takeaway:** VI is faster for this problem (14 vs 755 sweeps). Both produce the same optimal policy and values.

### 2. State-Value Heatmaps (`task5_state_value_heatmaps.png`)

These 6 panels show V*(s) — the expected total reward from each grid cell under the optimal policy. Green = high value (good), Red = low value (bad).

- **Battery=15, No rescues:** Full battery. Cells near rescue targets (0,4) and (4,0) are green (~37). The drone can reach either target easily.
- **Battery=7, No rescues:** Half battery. Values near the corners drop because the drone has less room to maneuver, but cells near targets stay green.
- **Battery=3, No rescues:** Low battery. Most cells turn orange/red. Only cells right next to a target or the charger remain positive.
- **Battery=1, No rescues:** Critical. Almost all cells are negative (red). The drone will die before reaching any target from most positions.
- **Battery=15, Target 0 rescued:** After rescuing T0 at (0,4), the green zone shifts entirely toward T1 at (4,0). The policy adapts to the remaining objective.
- **Battery=15, Target 1 rescued:** Mirror of the above — the green zone shifts toward T0 at (0,4).

**Takeaway:** Battery level drastically affects the value landscape. The policy dynamically re-plans based on which targets remain and how much battery is left.

### 3. Optimal Path Plot (`optimal_path.png`)

This plot shows the drone's step-by-step route when following the optimal policy, overlaid on the V*(s) heatmap.

- **Blue square** = start position (0,0)
- **Blue circles with numbers** = each step in order
- **Red star** = final position after both targets are rescued

The route is: Start (0,0) → move RIGHT along row 0 → rescue T0 at (0,4) → backtrack to charger at (2,2) to refuel → move down-left → rescue T1 at (4,0).

The drone avoids danger zones at (1,2), (3,0), and (4,3), and uses the charger strategically to ensure it has enough battery to complete the mission. Total reward: +36.

**Takeaway:** The DP-computed policy produces an efficient route that rescues the nearest target first, recharges, then rescues the second target.