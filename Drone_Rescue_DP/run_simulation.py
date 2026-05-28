"""Command-line simulation runner for the drone rescue environment.

This module wires together environment setup, action selection, logging,
rendering, and cleanup. It is intentionally kept separate from the environment
so the environment can remain focused on transition and reward rules.

Action modes:
    ``custom`` uses the hand-written ``CUSTOM_ACTIONS`` list.
    ``algorithm`` is a placeholder for future planning or learning logic.
    ``random`` samples actions from the Gymnasium action space.
"""

import logging
from datetime import datetime as dt
from pathlib import Path
import sys

try:
    from .environment import DroneRescueEnv
except ImportError:
    from environment import DroneRescueEnv

CUSTOM_ACTION_MODE = "custom"
ALGORITHM_ACTION_MODE = "algorithm"
RANDOM_ACTION_MODE = "random"
VALID_ACTION_MODES = {CUSTOM_ACTION_MODE, ALGORITHM_ACTION_MODE, RANDOM_ACTION_MODE}

# Default scripted policy used when running this file directly. Keeping this at
# module scope makes it easy to edit or import from tests.
CUSTOM_ACTIONS = [3, 3, 3, 3, 2, 1, 1, 2, 2, 1, 1, 2]

# All run artifacts are stored under logs/<timestamp>/ so each simulation keeps
# its own text log and rendered video.
LOG_ROOT = Path("logs")
LOG_SEPARATOR = "*" * 88


def create_log_directory():
    """Create a timestamped log directory for the simulation run.

    Each simulation run receives a separate folder to avoid overwriting logs or
    videos from previous runs.

    Returns:
        Path: Path to the created log directory.
    """
    folder_name = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = LOG_ROOT / folder_name
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def setup_logger(log_dir):
    """Configure logging to file and console for the simulation.

    The logger writes the same messages to the timestamped log file and to
    stdout. ``force=True`` resets existing logging configuration so repeated
    calls from notebooks or scripts do not duplicate messages.

    Args:
        log_dir (str): Directory where log files should be written.

    Returns:
        logging.Logger: Configured logger instance.
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


def initialize_environment(log_dir):
    """Create and reset the drone rescue environment.

    Rendering is enabled here because this runner is meant to produce a visual
    simulation. Training or unit tests can instantiate ``DroneRescueEnv`` with
    ``render_mode=None`` directly.

    Args:
        log_dir (str): Directory where environment logs or outputs should be stored.

    Returns:
        tuple: (env, observation, info) after the environment reset.
    """
    env = DroneRescueEnv(render_mode="human", log_dir=str(log_dir))
    observation, info = env.reset(seed=42)
    env.render()
    return env, observation, info


def log_initial_state(logger, observation, info):
    """Log the initial state of the simulation.

    The observation is logged along with the expanded info dictionary fields so
    debugging can compare what the agent sees against the internal state.

    Args:
        logger (logging.Logger): Logger instance used for simulation output.
        observation (Any): Initial observation from the environment.
        info (dict): Initial environment info dictionary.

    Returns:
        None
    """
    logger.info(
        "Initial Position: %s | Initial Battery: %s | Observation: %s",
        info["state"],
        info["battery_level"],
        observation,
    )


def generate_algorithm_actions(env):
    """Generate an action sequence using a planning or learning algorithm.

    This placeholder exists so future dynamic-programming, bandit, or
    reinforcement-learning logic can plug into the runner without changing the
    simulation loop.

    Args:
        env (DroneRescueEnv): The environment in which to generate actions.

    Returns:
        list[int]: A sequence of action integers for the agent to execute.
    """
    actions = []
    # =====================================================
    # TODO:
    # Add algorithm-generated action logic here
    # =====================================================
    return actions


def get_action_sequence(env, mode):
    """Return the chosen sequence of actions based on the selected mode.

    This function isolates mode handling from the simulation loop. Adding a new
    mode should only require defining a constant and extending this function.

    Args:
        env (DroneRescueEnv): The drone rescue environment instance.
        mode (str): The action mode to use (custom, algorithm, random).

    Returns:
        list[int]: Sequence of actions to execute.

    Raises:
        ValueError: If ``mode`` is not one of ``VALID_ACTION_MODES``.
    """
    if mode not in VALID_ACTION_MODES:
        valid_modes = ", ".join(sorted(VALID_ACTION_MODES))
        raise ValueError(
            f"Invalid action mode {mode!r}. Valid modes are: {valid_modes}."
        )

    if mode == CUSTOM_ACTION_MODE:
        return list(CUSTOM_ACTIONS)

    if mode == ALGORITHM_ACTION_MODE:
        return generate_algorithm_actions(env)

    return [env.action_space.sample() for _ in range(env.MAX_STEPS * 3)]


def execute_action(env, action):
    """Execute a single action in the environment.

    This wrapper keeps action execution easy to replace later if extra
    instrumentation, metrics, or debugging hooks are needed.

    Args:
        env (DroneRescueEnv): The environment instance.
        action (int): The action to execute.

    Returns:
        tuple: Observation, reward, terminated, truncated, and info from env.step().
    """
    return env.step(action)


def log_step(logger, info, reward, total_reward):
    """Log details for a single simulation step.

    The log format is deliberately stable and pipe-separated so it remains easy
    to scan by eye and can also be parsed by simple scripts if needed.

    Args:
        logger (logging.Logger): Logger instance for output.
        info (dict): Environment info returned by the last step.
        reward (float): Reward received from the last action.
        total_reward (float): Cumulative reward in the episode.

    Returns:
        None
    """
    logger.info(
        "Step: %s | Position: %s | Battery: %s | Reward: %s | Action: %s | Total Reward: %s",
        info["step_count"],
        info["state"],
        info["battery_level"],
        reward,
        info["action_taken"],
        total_reward,
    )


def handle_termination(logger, terminated, truncated, total_reward, info):
    """Handle episode termination logging and return whether simulation should stop.

    Gymnasium distinguishes true termination from truncation. This helper keeps
    that distinction visible in the logs and tells the caller whether to break
    out of the action loop.

    Args:
        logger (logging.Logger): Logger instance for output.
        terminated (bool): Whether the episode terminated normally.
        truncated (bool): Whether the episode was truncated by max steps.
        total_reward (float): Total reward accumulated.
        info (dict): Latest environment info dictionary.

    Returns:
        bool: True if the simulation should stop, otherwise False.
    """
    rescue_status = info.get("rescue_target_state", {})
    battery_level = info.get("battery_level", -1)
    rescue_status_count = sum(1 for status in rescue_status.values() if not status)

    logger.info(
        "Rescue Targets Rescued: %s | Rescue Status: %s",
        rescue_status_count,
        rescue_status,
    )
    logger.info(LOG_SEPARATOR)

    if terminated:
        termination_reason = get_termination_reason(
            rescue_status_count,
            len(rescue_status),
            battery_level,
        )
        logger.info(
            "Episode Terminated | Final Reward: %s | Reason: %s",
            total_reward,
            termination_reason,
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


def get_termination_reason(rescued_count, target_count, battery_level):
    """Return a readable termination reason for the current terminal state.

    Args:
        rescued_count (int): Number of targets already rescued.
        target_count (int): Total number of rescue targets.
        battery_level (int): Current battery level.

    Returns:
        str: Human-readable termination reason for logs.
    """
    if rescued_count == target_count:
        return "All Targets Rescued"
    if battery_level == 0:
        return "Battery Depleted"
    return "Environment/Agent Failure"


def simulation_loop(env, logger, action_mode):
    """Run the simulation loop until termination or action sequence exhaustion.

    The loop is intentionally linear: select actions, execute them one by one,
    log state changes, render each frame, and stop as soon as the episode ends.
    Any unexpected exception is logged with a traceback before being re-raised.

    Args:
        env (DroneRescueEnv): The environment instance.
        logger (logging.Logger): Logger for step and termination output.
        action_mode (str): Mode used to select actions.

    Returns:
        None
    """
    total_reward = 0.0
    actions = get_action_sequence(env, action_mode)

    if not actions:
        logger.warning("No actions available to execute.")
        return

    try:
        for action in actions:
            _, reward, terminated, truncated, info = execute_action(env, action)
            total_reward += reward

            logger.info("============================================================")
            log_step(logger, info, reward, total_reward)
            log_rescue_progress(logger, info)

            env.render()
            should_stop = handle_termination(
                logger,
                terminated,
                truncated,
                total_reward,
                info,
            )
            if should_stop:
                break
    except Exception:
        logger.exception("Simulation failed while executing an action.")
        raise


def log_rescue_progress(logger, info):
    """Log current rescue target progress.

    Args:
        logger (logging.Logger): Logger for simulation output.
        info (dict): Environment info containing rescue target status.
    """
    rescue_status = info.get("rescue_target_state", {})
    rescue_status_count = sum(1 for status in rescue_status.values() if not status)

    logger.info("Rescue Targets Rescued: %s", rescue_status_count)
    logger.info("Rescue Status: %s", rescue_status)


def cleanup_environment(env, logger):
    """Close resources and log the end of the simulation.

    Cleanup is centralized so normal completion and failure paths release the
    renderer consistently. ``logging.shutdown`` flushes and closes handlers.

    Args:
        env (DroneRescueEnv): The environment instance to close.
        logger (logging.Logger): Logger instance for shutdown messages.

    Returns:
        None
    """
    logger.info("------------- END SIMULATION -------------")
    try:
        env.close()
    except Exception:
        logger.exception("Failed to close environment cleanly.")
        raise
    finally:
        logging.shutdown()


def run_simulation(action_mode=CUSTOM_ACTION_MODE):
    """Run the full drone rescue simulation workflow.

    This is the high-level entry point used by the ``__main__`` block. It keeps
    setup, execution, error logging, and cleanup in one predictable flow.

    Args:
        action_mode (str): The action selection mode to use.

    Returns:
        None
    """
    log_dir = create_log_directory()
    logger = setup_logger(log_dir)
    env = None

    logger.info("------------- START SIMULATION -------------")
    try:
        env, observation, info = initialize_environment(log_dir)
        log_initial_state(logger, observation, info)
        simulation_loop(env, logger, action_mode)
    except Exception:
        logger.exception("Simulation stopped because of an unexpected error.")
        raise
    finally:
        if env is not None:
            cleanup_environment(env, logger)


# =========================================================
# CUSTOM ACTIONS
# =========================================================
# 0 -> UP
# 1 -> DOWN
# 2 -> LEFT
# 3 -> RIGHT
# 4 -> HOVER
# =========================================================

if __name__ == "__main__":
    # CUSTOM_ACTIONS = [3,3,3,2,3,2,3,2,3,2,3,2,3]

    run_simulation(action_mode=CUSTOM_ACTION_MODE)

    # OR

    # run_simulation(action_mode=ALGORITHM_ACTION_MODE)

    # OR

    # run_simulation(action_mode=RANDOM_ACTION_MODE)
