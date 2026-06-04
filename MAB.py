"""
=============================================================
BIRLA INSTITUTE OF TECHNOLOGY AND SCIENCE, PILANI
WORK INTEGRATED LEARNING PROGRAMMES DIVISION

Deep Reinforcement Learning - Lab Assignment 1
Part #1: Multi-Armed Bandit (MAB)
Team Number: 191

Tasks Implemented:
  Task 1 - Dataset Design (Synthetic Patient-Treatment Environment)
  Task 2 - Immediate Exploitation Strategy (Greedy after warm-up)

Group Number (G): 191
=============================================================
"""

# ─────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────
import os                         # For building relative output paths
import random                    # For reproducible random sampling
import numpy as np               # For numerical operations and seeding
import pandas as pd              # For structured dataset creation and display
import matplotlib.pyplot as plt  # For plotting cumulative reward graphs

# Resolve the directory this script lives in so plots are saved next to it
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────
# SECTION 1: GROUP PARAMETERS
# ─────────────────────────────────────────────────────────────

# Set group number
G = 191

# Set random seeds for reproducibility (as required by assignment)
random.seed(G)
np.random.seed(G)

# ── 1.1 Number of Medicines ──────────────────────────────────
# Formula: K = (G mod 3) + 5
K = (G % 3) + 5

# ── 1.2 Hidden Success Probabilities ────────────────────────
# Formula: P_i = 0.4 + ((G + i) mod 6) * 0.07  for i in {0, 1, ..., K-1}
hidden_probs = [0.4 + ((G + i) % 6) * 0.07 for i in range(K)]

# ── Display Group Parameters ─────────────────────────────────
print("=" * 60)
print("       ADAPTIVE TREATMENT RECOMMENDATION SYSTEM")
print("              Multi-Armed Bandit (MAB)")
print("=" * 60)
print(f"\n  Group Number (G)          : {G}")
print(f"  Total Medicines (K)       : {K}")
print(f"  Formula for K             : ({G} mod 3) + 5 = {G % 3} + 5 = {K}")
print(f"\n  Hidden Success Probabilities per Medicine:")
print(f"  {'Medicine':<12} {'Formula':<30} {'P_i':>6}")
print(f"  {'-'*50}")
for i in range(K):
    formula_str = f"0.4 + (({G}+{i}) mod 6)*0.07 = 0.4 + {(G+i)%6}*0.07"
    print(f"  Medicine {i:<3}  {formula_str:<30}  {hidden_probs[i]:.4f}")
print()


# ─────────────────────────────────────────────────────────────
# SECTION 2: DATASET CREATION
# ─────────────────────────────────────────────────────────────

def compute_severity(patient_id):
    """
    Compute the disease severity score for a given patient.

    Formula: Severity = (patient_id mod 5) + 1
    Returns values from 1 (mild) to 5 (critical).

    Args:
        patient_id (int): Index of the patient (0 to 999).

    Returns:
        int: Severity score between 1 and 5.
    """
    return (patient_id % 5) + 1


def get_clinical_outcome(medicine_index, hidden_probs):
    """
    Simulate binary clinical outcome for a patient given a medicine.

    The outcome is sampled from a Bernoulli distribution using the
    medicine's hidden success probability.

    Args:
        medicine_index (int): Index of the assigned medicine (0 to K-1).
        hidden_probs (list): List of hidden success probabilities per medicine.

    Returns:
        int: 1 if patient recovers, 0 if not.
    """
    return int(np.random.rand() < hidden_probs[medicine_index])


def compute_utility(clinical_outcome, severity):
    """
    Compute the utility (reward) score based on clinical outcome and severity.

    Formula: UtilityScore = clinical_outcome * (1 - severity / 10)
    Higher severity reduces the benefit even when recovery occurs.

    Examples:
        - Recovered, severity=1 → reward = 1 * (1 - 0.1) = 0.9
        - Recovered, severity=5 → reward = 1 * (1 - 0.5) = 0.5
        - Not recovered         → reward = 0

    Args:
        clinical_outcome (int): 1 (recovered) or 0 (not recovered).
        severity (int): Severity score (1–5).

    Returns:
        float: Utility score (reward) for this patient-treatment pair.
    """
    return clinical_outcome * (1 - severity / 10)


def create_base_dataset(num_patients=1000):
    """
    Generate the base synthetic patient dataset with patient_id and severity_score.

    Only static columns are created here. Dynamic columns (assigned_medicine,
    clinical_outcome, utility_score) are populated during algorithm execution.

    Args:
        num_patients (int): Total number of patients to simulate. Default is 1000.

    Returns:
        pd.DataFrame: DataFrame with patient_id and severity_score columns.
    """
    records = []
    for pid in range(num_patients):
        severity = compute_severity(pid)
        records.append({
            "patient_id": pid,
            "severity_score": severity
        })
    return pd.DataFrame(records)


# Create the base dataset
base_df = create_base_dataset(num_patients=1000)

print("=" * 60)
print("  TASK 1: DATASET DESIGN")
print("=" * 60)
print(f"\n  Total patient records created : {len(base_df)}")
print(f"  Severity distribution (1=mild to 5=critical):")
print(base_df["severity_score"].value_counts().sort_index().to_string())
print(f"\n  First 10 rows of base dataset (before algorithm runs):")
print(base_df.head(10).to_string(index=False))
print()


# ─────────────────────────────────────────────────────────────
# SECTION 3: TASK 2 – IMMEDIATE EXPLOITATION STRATEGY
# ─────────────────────────────────────────────────────────────

"""
Strategy: Greedy Exploitation after Warm-Up

Step 1 (Warm-Up / Exploration):
    Test each of the K medicines exactly 10 times (in round-robin order)
    to gather initial statistics. Total warm-up patients = K * 10.

Step 2 (Exploitation):
    Identify the best-performing medicine based on average clinical_outcome
    (not utility) during warm-up, then prescribe ONLY that medicine to all
    remaining patients.

This models a "commit-and-exploit" hospital administrator policy.
"""

WARMUP_TRIALS_PER_MEDICINE = 10   # Each medicine tested this many times first
NUM_PATIENTS = 1000                # Total simulation length


def run_immediate_exploitation(base_df, hidden_probs, K,
                                warmup_per_arm=10, seed=G):
    """
    Run the Immediate Exploitation (Greedy after Warm-Up) strategy.

    Phase 1 – Warm-Up:
        Each medicine is tested exactly `warmup_per_arm` times in round-robin.
        Clinical outcomes are recorded to estimate each medicine's success rate.

    Phase 2 – Exploitation:
        The medicine with the highest average success rate from warm-up is
        selected and used exclusively for all remaining patients.

    Args:
        base_df (pd.DataFrame): Base dataset with patient_id and severity_score.
        hidden_probs (list): Hidden success probabilities for each medicine.
        K (int): Total number of available medicines.
        warmup_per_arm (int): Number of initial trials per medicine.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Full dataset with assigned_medicine, clinical_outcome,
                      utility_score columns populated.
        dict: Summary statistics including best medicine found and rewards.
    """

    # Re-seed before running to ensure reproducibility
    np.random.seed(seed)

    # Copy the base dataset to avoid modifying the original
    df = base_df.copy()

    # Initialise dynamic columns as empty
    df["assigned_medicine"] = -1       # Which medicine was prescribed
    df["clinical_outcome"]  = -1       # 1 = recovered, 0 = not
    df["utility_score"]     = 0.0      # Reward = outcome * (1 - severity/10)

    # ── Tracking variables ────────────────────────────────────
    arm_success_count = np.zeros(K)    # Total recoveries per medicine
    arm_trial_count   = np.zeros(K)    # Total prescriptions per medicine
    cumulative_reward = []             # Running total of utility scores
    total_reward      = 0.0

    best_medicine     = None           # Best arm found after warm-up
    best_avg          = -1.0          # Best average clinical outcome

    warmup_total = K * warmup_per_arm  # Total warm-up patients

    print("=" * 60)
    print("  TASK 2: IMMEDIATE EXPLOITATION STRATEGY")
    print("=" * 60)
    print(f"\n  Strategy   : Greedy Exploitation after Warm-Up")
    print(f"  Warm-up    : {warmup_per_arm} trials per medicine × {K} medicines"
          f" = {warmup_total} patients")
    print(f"  Exploitation: Remaining {NUM_PATIENTS - warmup_total} patients")
    print()

    # ── Phase 1: Warm-Up (Round-Robin Exploration) ─────────────
    print("  [Phase 1] Warm-Up Exploration:")
    for patient_idx in range(warmup_total):
        pid      = df.loc[patient_idx, "patient_id"]
        severity = df.loc[patient_idx, "severity_score"]

        # Assign medicines in round-robin order: 0,1,2,...,K-1,0,1,...
        medicine = patient_idx % K

        # Simulate clinical outcome using hidden probability
        outcome = get_clinical_outcome(medicine, hidden_probs)

        # Compute reward (utility) based on outcome and severity
        utility = compute_utility(outcome, severity)

        # Record in dataset
        df.at[patient_idx, "assigned_medicine"] = medicine
        df.at[patient_idx, "clinical_outcome"]  = outcome
        df.at[patient_idx, "utility_score"]     = utility

        # Update arm statistics (use clinical_outcome per assignment spec)
        arm_success_count[medicine] += outcome
        arm_trial_count[medicine]   += 1

        # Track cumulative reward
        total_reward += utility
        cumulative_reward.append(total_reward)

    # Determine best medicine after warm-up
    arm_avg = arm_success_count / arm_trial_count  # Average success rate per arm
    best_medicine = int(np.argmax(arm_avg))
    best_avg      = arm_avg[best_medicine]

    print(f"  Warm-up complete. Medicine statistics:")
    print(f"  {'Medicine':<10} {'Trials':>8} {'Successes':>10} {'Avg Success':>12}"
          f" {'True P_i':>10}")
    for i in range(K):
        marker = " ← BEST" if i == best_medicine else ""
        print(f"  Medicine {i:<2}  {int(arm_trial_count[i]):>8}"
              f"  {int(arm_success_count[i]):>10}  {arm_avg[i]:>11.4f}"
              f"  {hidden_probs[i]:>10.4f}{marker}")
    print()
    print(f"  Best medicine identified: Medicine {best_medicine}"
          f" (avg success = {best_avg:.4f})")
    print()

    # ── Phase 2: Pure Exploitation ─────────────────────────────
    print(f"  [Phase 2] Exploiting Medicine {best_medicine} exclusively:")
    for patient_idx in range(warmup_total, NUM_PATIENTS):
        pid      = df.loc[patient_idx, "patient_id"]
        severity = df.loc[patient_idx, "severity_score"]

        # Always use the best medicine found in warm-up
        medicine = best_medicine

        # Simulate outcome
        outcome = get_clinical_outcome(medicine, hidden_probs)

        # Compute utility
        utility = compute_utility(outcome, severity)

        # Record in dataset
        df.at[patient_idx, "assigned_medicine"] = medicine
        df.at[patient_idx, "clinical_outcome"]  = outcome
        df.at[patient_idx, "utility_score"]     = utility

        # Update stats (optional tracking)
        arm_success_count[medicine] += outcome
        arm_trial_count[medicine]   += 1

        # Track cumulative reward
        total_reward += utility
        cumulative_reward.append(total_reward)

    # ── Summary ───────────────────────────────────────────────
    summary = {
        "strategy"           : "Immediate Exploitation (Greedy)",
        "best_medicine"      : best_medicine,
        "best_medicine_prob" : hidden_probs[best_medicine],
        "true_optimal_med"   : int(np.argmax(hidden_probs)),
        "true_optimal_prob"  : max(hidden_probs),
        "total_reward"       : round(total_reward, 4),
        "cumulative_reward"  : cumulative_reward,
        "arm_trials"         : arm_trial_count,
        "arm_successes"      : arm_success_count,
    }

    return df, summary


# ── Run the strategy ─────────────────────────────────────────
result_df, summary = run_immediate_exploitation(
    base_df, hidden_probs, K,
    warmup_per_arm=WARMUP_TRIALS_PER_MEDICINE, seed=G
)

# ── Print Results ─────────────────────────────────────────────
print("  Results:")
print(f"  {'─'*45}")
print(f"  Best medicine found by algorithm : Medicine {summary['best_medicine']}"
      f" (P = {summary['best_medicine_prob']:.4f})")
print(f"  True optimal medicine            : Medicine {summary['true_optimal_med']}"
      f" (P = {summary['true_optimal_prob']:.4f})")
if summary['best_medicine'] == summary['true_optimal_med']:
    print(f"  ✓ Algorithm correctly identified the optimal medicine!")
else:
    print(f"  ✗ Algorithm chose a sub-optimal medicine due to warm-up variance.")
print(f"  Total cumulative utility reward  : {summary['total_reward']:.4f}")
print()

# ── Show first 10 rows of completed dataset ──────────────────
print("  First 10 rows of fully populated dataset:")
print(result_df.head(10).to_string(index=False))
print()

# ── Show breakdown of medicine assignments ───────────────────
print("  Medicine Assignment Breakdown:")
print(result_df["assigned_medicine"].value_counts().sort_index().to_string())
print()

# ── Final cumulative reward ───────────────────────────────────
print("  Cumulative Reward at each 100-patient milestone:")
for milestone in range(100, NUM_PATIENTS + 1, 100):
    print(f"    After patient {milestone:>4}: "
          f"{summary['cumulative_reward'][milestone-1]:.4f}")
print()


# ─────────────────────────────────────────────────────────────
# SECTION 4: CUMULATIVE REWARD PLOT (Task 2 standalone)
# ─────────────────────────────────────────────────────────────

def plot_exploitation_reward(summary, K, warmup_total):
    """
    Plot the Cumulative Reward vs Number of Patients for the
    Immediate Exploitation strategy.

    A vertical dashed line marks the end of the warm-up phase.
    Annotations indicate which medicine was selected post-warm-up.

    Args:
        summary (dict): Output from run_immediate_exploitation().
        K (int): Number of medicines.
        warmup_total (int): Number of patients used in warm-up phase.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    patients = list(range(1, NUM_PATIENTS + 1))
    rewards  = summary["cumulative_reward"]

    ax.plot(patients, rewards, color="#1a73e8", linewidth=2,
            label="Immediate Exploitation (Greedy)")

    # Mark end of warm-up phase
    ax.axvline(x=warmup_total, color="orange", linestyle="--", linewidth=1.5,
               label=f"End of Warm-Up (patient {warmup_total})")

    ax.fill_between(patients[:warmup_total],
                    rewards[:warmup_total], alpha=0.15, color="orange",
                    label="Warm-Up Phase")
    ax.fill_between(patients[warmup_total:],
                    rewards[warmup_total:], alpha=0.10, color="#1a73e8",
                    label="Exploitation Phase")

    # Annotate best medicine
    ax.annotate(
        f"Exploiting Medicine {summary['best_medicine']}\n"
        f"P = {summary['best_medicine_prob']:.4f}",
        xy=(warmup_total + 50, rewards[warmup_total + 50]),
        xytext=(warmup_total + 120, rewards[warmup_total + 50] - 20),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=9, color="#333"
    )

    ax.set_xlabel("Number of Patients", fontsize=12)
    ax.set_ylabel("Cumulative Utility Reward", fontsize=12)
    ax.set_title(
        f"Task 2: Immediate Exploitation Strategy – Cumulative Reward\n"
        f"Group {G} | K={K} Medicines | Warm-Up={warmup_total} patients",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    _plot_path = os.path.join(_SCRIPT_DIR, "Team_191_Task2_Exploitation_Reward.png")
    plt.savefig(_plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Plot saved: {_plot_path}")


warmup_total = K * WARMUP_TRIALS_PER_MEDICINE
plot_exploitation_reward(summary, K, warmup_total)


# ─────────────────────────────────────────────────────────────
# END OF SUBMISSION – Team 191 | Tasks 1 & 2
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Submission: Team 191 | Part 1 MAB | Tasks 1 & 2 Complete")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# SECTION 5: TASK 3 – CONTROLLED CLINICAL TRIAL (Epsilon-Greedy)
# ─────────────────────────────────────────────────────────────

def run_epsilon_greedy(base_df, hidden_probs, K, epsilon, seed=G):
    """
    Run the Epsilon-Greedy strategy.
    
    - With probability `epsilon`, explore by choosing a random medicine.
    - With probability `1 - epsilon`, exploit by choosing the medicine 
      with the highest average clinical success rate so far.
    """
    
    np.random.seed(seed)
    random.seed(seed)
    
    
    df = base_df.copy()
    
    # Initialize dynamic columns
    df["assigned_medicine"] = -1
    df["clinical_outcome"]  = -1
    df["utility_score"]     = 0.0
    
    # Tracking variables
    arm_success_count = np.zeros(K)
    arm_trial_count   = np.zeros(K)
    cumulative_reward = []
    total_reward      = 0.0
    
    # Run the simulation for all 1000 patients
    for patient_idx in range(len(df)):
        severity = df.loc[patient_idx, "severity_score"]
        
        # ── 1. MAKE A CHOICE (Explore vs Exploit) ─────────────
        if np.random.rand() < epsilon:
            # EXPLORE: Pick a completely random medicine
            medicine = random.randint(0, K - 1)
        else:
            # EXPLOIT: Pick the best known medicine
            # Calculate averages safely (avoiding divide-by-zero errors)
            arm_avg = np.zeros(K)
            for i in range(K):
                if arm_trial_count[i] > 0:
                    arm_avg[i] = arm_success_count[i] / arm_trial_count[i]
            
            # Find the maximum average
            max_avg = np.max(arm_avg)
            
            # To avoid bias, if multiple medicines tie for the highest average 
            # (which happens a lot at the very beginning), pick randomly among the winners.
            best_medicines = np.flatnonzero(arm_avg == max_avg)
            medicine = np.random.choice(best_medicines)
            
        # ── 2. SIMULATE OUTCOME & REWARD ──────────────────────
        outcome = get_clinical_outcome(medicine, hidden_probs)
        utility = compute_utility(outcome, severity)
        
        # ── 3. UPDATE DATASET ─────────────────────────────────
        df.at[patient_idx, "assigned_medicine"] = medicine
        df.at[patient_idx, "clinical_outcome"]  = outcome
        df.at[patient_idx, "utility_score"]     = utility
        
        # ── 4. UPDATE INTERNAL TRACKERS ───────────────────────
        arm_success_count[medicine] += outcome
        arm_trial_count[medicine]   += 1
        
        total_reward += utility
        cumulative_reward.append(total_reward)

    # ── Summary Dictionary ───────────────────────────────────
    summary = {
        "strategy"          : f"Epsilon-Greedy (eps={epsilon})",
        "total_reward"      : round(total_reward, 4),
        "cumulative_reward" : cumulative_reward,
        "arm_trials"        : arm_trial_count,
        "arm_successes"     : arm_success_count,
    }
    
    return df, summary


# ── Run the required scenarios for Task 3 ────────────────────
print("=" * 60)
print("  TASK 3: CONTROLLED CLINICAL TRIAL STRATEGY (Epsilon-Greedy)")
print("=" * 60)

# Scenario A: 10% Exploration (The main requirement)
df_eps_10, sum_eps_10 = run_epsilon_greedy(base_df, hidden_probs, K, epsilon=0.10, seed=G)
print(f"  [10% Exploration] Total Reward: {sum_eps_10['total_reward']:.4f}")

# Scenario B: 1% Exploration (For analysis)
df_eps_01, sum_eps_01 = run_epsilon_greedy(base_df, hidden_probs, K, epsilon=0.01, seed=G)
print(f"  [ 1% Exploration] Total Reward: {sum_eps_01['total_reward']:.4f}")

# Scenario C: 50% Exploration (For analysis)
df_eps_50, sum_eps_50 = run_epsilon_greedy(base_df, hidden_probs, K, epsilon=0.50, seed=G)
print(f"  [50% Exploration] Total Reward: {sum_eps_50['total_reward']:.4f}\n")

# Show how the 10% model distributed its trials across the 7 medicines
print("  Trial Distribution for 10% Exploration model:")
for i in range(K):
    print(f"  Medicine {i}: {int(sum_eps_10['arm_trials'][i])} times prescribed")


# ─────────────────────────────────────────────────────────────
# SECTION 6: TASK 4 – CONFIDENCE-BASED STRATEGY (UCB1)
# ─────────────────────────────────────────────────────────────

def run_ucb1(base_df, hidden_probs, K, seed=G):
    """
    Run the UCB1 (Upper Confidence Bound) strategy.
    
    Phase 1 (Initialization): Play each arm exactly once to avoid divide-by-zero.
    Phase 2 (UCB): Pick the arm that maximizes the UCB1 formula:
                   UCB_i = avg_success_rate + sqrt(2 * ln(t) / N_i)
    """
    np.random.seed(seed)
    random.seed(seed)
    
    df = base_df.copy()
    
    df["assigned_medicine"] = -1
    df["clinical_outcome"]  = -1
    df["utility_score"]     = 0.0
    
    arm_success_count = np.zeros(K)
    arm_trial_count   = np.zeros(K)
    cumulative_reward = []
    total_reward      = 0.0
    
    for patient_idx in range(len(df)):
        severity = df.loc[patient_idx, "severity_score"]
        
        # 't' is the current time step (starts at 1 to avoid ln(0))
        t = patient_idx + 1
        
        # ── 1. MAKE A CHOICE ──────────────────────────────────
        # Check if there are any medicines we haven't tried yet (Phase 1)
        untested_arms = np.where(arm_trial_count == 0)[0]
        
        if len(untested_arms) > 0:
            # If so, pick the first untested medicine
            medicine = untested_arms[0]
        else:
            # Phase 2: Apply the UCB1 formula
            ucb_scores = np.zeros(K)
            
            for i in range(K):
                # Calculate average success rate (Exploitation)
                avg_success = arm_success_count[i] / arm_trial_count[i]
                
                # Calculate uncertainty bonus (Exploration)
                # Note: We use math.log for natural logarithm (ln)
                uncertainty_bonus = np.sqrt((2 * np.log(t)) / arm_trial_count[i])
                
                ucb_scores[i] = avg_success + uncertainty_bonus
            
            # Find the medicine with the highest UCB score
            max_ucb = np.max(ucb_scores)
            best_medicines = np.flatnonzero(ucb_scores == max_ucb)
            medicine = np.random.choice(best_medicines)
            
        # ── 2. SIMULATE OUTCOME & REWARD ──────────────────────
        outcome = get_clinical_outcome(medicine, hidden_probs)
        utility = compute_utility(outcome, severity)
        
        # ── 3. UPDATE DATASET ─────────────────────────────────
        df.at[patient_idx, "assigned_medicine"] = medicine
        df.at[patient_idx, "clinical_outcome"]  = outcome
        df.at[patient_idx, "utility_score"]     = utility
        
        # ── 4. UPDATE INTERNAL TRACKERS ───────────────────────
        # Per instructions, we use clinical_outcome to update bandit statistics
        arm_success_count[medicine] += outcome
        arm_trial_count[medicine]   += 1
        
        total_reward += utility
        cumulative_reward.append(total_reward)

    summary = {
        "strategy"          : "UCB1",
        "total_reward"      : round(total_reward, 4),
        "cumulative_reward" : cumulative_reward,
        "arm_trials"        : arm_trial_count,
        "arm_successes"     : arm_success_count,
    }
    
    return df, summary


# ── Run the strategy for Task 4 ──────────────────────────────
print("=" * 60)
print("  TASK 4: CONFIDENCE-BASED STRATEGY (UCB1)")
print("=" * 60)

df_ucb1, sum_ucb1 = run_ucb1(base_df, hidden_probs, K, seed=G)

print(f"  [UCB1] Total Cumulative Reward: {sum_ucb1['total_reward']:.4f}\n")
print("  Trial Distribution for UCB1 model:")
for i in range(K):
    print(f"  Medicine {i}: {int(sum_ucb1['arm_trials'][i])} times prescribed")

# ─────────────────────────────────────────────────────────────
# SECTION 7: TASK 5 – COMPARATIVE ANALYSIS
# ─────────────────────────────────────────────────────────────

def plot_comparative_analysis(sum_greedy, sum_eps, sum_ucb, G):
    """
    Generate a Cumulative Reward vs. Number of Patients graph comparing
    all implemented strategies over 1000 iterations.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    patients = list(range(1, NUM_PATIENTS + 1))
    
    # Plot Task 2: Immediate Exploitation (Greedy)
    ax.plot(patients, sum_greedy["cumulative_reward"], 
            color="#e53935", linewidth=2, label="Task 2: Immediate Exploitation (Greedy)")
    
    # Plot Task 3: Epsilon-Greedy (10%)
    ax.plot(patients, sum_eps["cumulative_reward"], 
            color="#43a047", linewidth=2, label="Task 3: Epsilon-Greedy (10%)")
    
    # Plot Task 4: UCB1
    ax.plot(patients, sum_ucb["cumulative_reward"], 
            color="#1e88e5", linewidth=2, label="Task 4: UCB1 (Confidence-Based)")
    
    # Formatting
    ax.set_xlabel("Number of Patients", fontsize=12)
    ax.set_ylabel("Cumulative Utility Reward", fontsize=12)
    ax.set_title(
        f"Task 5: Comparative Analysis of MAB Strategies\nGroup {G} | 1000 Patients", 
        fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    _plot_path = os.path.join(_SCRIPT_DIR, "Team_191_Task5_Comparative_Analysis.png")
    plt.savefig(_plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Comparative Plot saved: {_plot_path}")

print("=" * 60)
print("  TASK 5: COMPARATIVE ANALYSIS")
print("=" * 60)
# Make sure the variable names match what your previous cells output!
plot_comparative_analysis(summary, sum_eps_10, sum_ucb1, G)    