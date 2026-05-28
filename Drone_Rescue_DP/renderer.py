"""Pygame renderer for the drone rescue environment.

The renderer is intentionally separate from the Gymnasium environment so the
simulation rules can run headlessly during training or tests. This module owns
all visual layout decisions, status panels, legend drawing, path history, frame
capture, and MP4 export.
"""

import imageio
import os

import numpy as np
import pygame


class GridRenderer:
    """Visual renderer for the drone rescue grid and simulation status.

    The renderer draws three major areas:
        1. The rescue grid and the drone's historical path.
        2. A status panel with state, battery, reward, and last action.
        3. A legend and rescue-target progress panel.

    Frames are stored in memory and written to ``simulation.mp4`` when
    ``save_video`` is called.
    """

    # Fixed layout constants keep the visual output stable from frame to frame.
    CELL_SIZE = 95
    INFO_PANEL_WIDTH = 820
    WIDTH = 1296
    HEIGHT = 736

    # Shared colors for panels and labels.
    BG_COLOR = (8, 12, 28)
    BOX_COLOR = (36, 38, 52)
    BORDER_COLOR = (98, 102, 145)
    TEXT_COLOR = (240, 240, 245)
    WHITE = (255, 255, 255)

    # Cell colors mirror the environment cell-type constants.
    COLORS = {
        -1: (90, 90, 100),
        0: (238, 238, 238),
        1: (45, 200, 70),
        2: (255, 220, 40),
        3: (235, 70, 70),
        4: (40, 110, 255),
        5: (170, 60, 255),
    }

    def __init__(self, rows=5, cols=5, fps=20, log_dir="logs"):
        """Initialize the renderer and Pygame display.

        Pygame setup happens here rather than in the environment so rendering
        errors are localized to this module. The constructor validates basic
        layout settings before opening the display window.

        Args:
            rows (int): Number of grid rows.
            cols (int): Number of grid columns.
            fps (int): Frames per second for video output.
            log_dir (str): Directory for saving rendered output.
        """
        if rows <= 0 or cols <= 0:
            raise ValueError("Renderer rows and columns must be positive integers.")
        if fps <= 0:
            raise ValueError("Renderer FPS must be a positive integer.")

        try:
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        except pygame.error as error:
            raise RuntimeError(f"Could not initialize the Pygame display: {error}") from error

        self.rows = rows
        self.cols = cols
        self.fps = fps
        self.log_dir = log_dir

        pygame.display.set_caption("Drone Rescue Environment")

        # Fonts are centralized here so text styling stays consistent across
        # status panels, headings, and legend items.
        self.font = pygame.font.SysFont("Segoe UI", 22)
        self.small_font = pygame.font.SysFont("Segoe UI", 18)
        self.title_font = pygame.font.SysFont("Segoe UI", 40, bold=True)
        self.heading_font = pygame.font.SysFont("Segoe UI", 30, bold=True)

        # frames stores captured images for MP4 export. path_history stores grid
        # coordinates already visited by the drone.
        self.frames = []
        self.path_history = []

    def draw_box(self, x, y, width, height, radius=20):
        """Draw a rounded overlay box for the UI panels.

        This helper keeps panel styling consistent and avoids repeating border
        drawing code throughout the render method.

        Args:
            x (int): X coordinate for the box.
            y (int): Y coordinate for the box.
            width (int): Width of the box.
            height (int): Height of the box.
            radius (int): Border radius of the box corners.

        Returns:
            None
        """
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, self.BOX_COLOR, rect, border_radius=radius)
        pygame.draw.rect(self.screen, self.BORDER_COLOR, rect, 2, border_radius=radius)

    def draw_legend_item(self, x, y, color, text):
        """Draw a single legend entry with a color dot and text label.

        Legend items use circles instead of full cell squares so the legend
        stays compact while still matching the grid colors.

        Args:
            x (int): X coordinate of the legend item.
            y (int): Y coordinate of the legend item.
            color (tuple): RGB color tuple for the legend dot.
            text (str): Label text for the legend item.

        Returns:
            None
        """
        pygame.draw.circle(self.screen, color, (x, y + 11), 8)
        legend_surface = self.small_font.render(text, True, (230, 230, 235))
        self.screen.blit(legend_surface, (x + 24, y))

    def render(
        self,
        grid,
        state,
        battery_level,
        step_count,
        cumulative_reward,
        rescue_target_state,
        action_taken,
    ):
        """Render the current simulation frame and update the display.

        The renderer expects already-valid environment state and then performs
        its own shape/type checks before drawing. Each rendered frame is copied
        into ``self.frames`` for later video export.

        Args:
            grid (np.ndarray): The grid values representing the environment.
            state (tuple): Current drone position as (row, col).
            battery_level (int): Current battery remaining.
            step_count (int): Number of steps taken so far.
            cumulative_reward (float): Total accumulated reward.
            rescue_target_state (dict): Mapping of rescue target positions to active status.
            action_taken (str): Last action taken by the drone.

        Returns:
            None
        """
        self._validate_render_inputs(grid, state, rescue_target_state)

        self.screen.fill(self.BG_COLOR)
        padding = 10
        grid_total_height = self.rows * self.CELL_SIZE
        vertical_offset = (self.HEIGHT - grid_total_height) // 2 + 20

        # Draw the grid first so path markers and the drone appear above cells.
        for row in range(self.rows):
            for col in range(self.cols):
                cell_value = grid[row, col]
                color = self.COLORS.get(cell_value, self.COLORS[0])
                x = col * self.CELL_SIZE + padding
                y = vertical_offset + row * self.CELL_SIZE + padding
                cell_size = self.CELL_SIZE - (padding * 2)
                rect = pygame.Rect(x, y, cell_size, cell_size)
                pygame.draw.rect(self.screen, color, rect, border_radius=18)
                pygame.draw.rect(self.screen, self.WHITE, rect, 2, border_radius=18)

        # Path history gives a simple visual trace of where the drone has been.
        for previous_state in self.path_history:
            prev_x, prev_y = self._cell_center(previous_state, vertical_offset, padding)
            pygame.draw.circle(self.screen, (130, 130, 130), (prev_x, prev_y), 7)

        # The drone is drawn after the path so the current position is prominent.
        drone_x, drone_y = self._cell_center(state, vertical_offset, padding)
        pygame.draw.circle(self.screen, (0, 255, 255), (drone_x, drone_y), 30)

        # Status panel summarizes the latest environment state.
        panel_x = self.cols * self.CELL_SIZE + 45
        title_surface = self.title_font.render("DRONE STATUS", True, self.WHITE)
        self.screen.blit(title_surface, (panel_x, 36))

        self.draw_box(panel_x, 120, 360, 340)
        status_lines = [
            ("State", state),
            ("Step", step_count),
            ("Battery", battery_level),
            ("Reward", round(cumulative_reward, 2)),
            ("Action", action_taken or "None"),
        ]

        line_spacing = 58
        content_height = len(status_lines) * line_spacing
        start_y = 120 + (340 - content_height) // 2 + 10
        label_x = panel_x + 48
        value_x = panel_x + 185

        # Render label/value pairs with a fixed colon position for scanability.
        for i, (label, value) in enumerate(status_lines):
            y = start_y + i * line_spacing
            label_surface = self.font.render(f"{label}", True, self.TEXT_COLOR)
            colon_surface = self.font.render(":", True, self.TEXT_COLOR)
            value_surface = self.font.render(f"{value}", True, self.TEXT_COLOR)
            self.screen.blit(label_surface, (label_x, y))
            self.screen.blit(colon_surface, (label_x + 110, y))
            self.screen.blit(value_surface, (value_x, y))

        legend_x = panel_x + 385
        self.draw_box(legend_x, 120, 360, 340)
        legend_title = self.heading_font.render("LEGEND", True, self.WHITE)
        self.screen.blit(legend_title, (legend_x + 24, 145))

        left_x = legend_x + 30
        right_x = legend_x + 205

        # The legend is split into two columns so it fits inside the panel.
        self.draw_legend_item(left_x, 215, (45, 200, 70), "Start")
        self.draw_legend_item(left_x, 265, (238, 238, 238), "Free Cell")
        self.draw_legend_item(left_x, 315, (255, 220, 40), "Wind Zone")
        self.draw_legend_item(left_x, 365, (235, 70, 70), "Danger Zone")

        self.draw_legend_item(right_x, 215, (40, 110, 255), "Charging")
        self.draw_legend_item(right_x, 265, (170, 60, 255), "Rescue")
        self.draw_legend_item(right_x, 315, (90, 90, 100), "Blocked")
        self.draw_legend_item(right_x, 365, (0, 255, 255), "Drone")

        self.draw_box(panel_x, 500, 740, 165)
        rescue_title = self.heading_font.render("RESCUE TARGETS", True, self.WHITE)
        self.screen.blit(rescue_title, (panel_x + 24, 515))

        # Each target displays its grid coordinate and whether it still needs
        # rescue. The environment owns the target state; the renderer only reads.
        for i, (target, active) in enumerate(rescue_target_state.items()):
            target_surface = self.font.render(f"{target}", True, (240, 240, 240))
            status = "ACTIVE" if active else "DONE"
            status_color = (0, 220, 100) if active else (255, 140, 0)
            status_surface = self.font.render(status, True, status_color)
            y_pos = 570 + i * 48
            self.screen.blit(target_surface, (panel_x + 36, y_pos))
            self.screen.blit(status_surface, (panel_x + 230, y_pos))

        pygame.display.flip()
        pygame.time.delay(2000)

        # Pygame surfaces are width-first; image/video libraries expect
        # height-first arrays, so transpose before storing the frame.
        frame = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(frame, (1, 0, 2))

        # Repeat each frame so the final video is readable instead of flashing
        # through states too quickly.
        for _ in range(120):
            self.frames.append(frame.copy())

        if state not in self.path_history:
            self.path_history.append(state)

    def _cell_center(self, state, vertical_offset, padding):
        """Return the pixel center for a grid state.

        Args:
            state (tuple[int, int]): Grid coordinate as ``(row, column)``.
            vertical_offset (int): Top offset used to vertically center the grid.
            padding (int): Inner cell padding.

        Returns:
            tuple[int, int]: Pixel coordinate for the cell center.
        """
        row, col = state
        cell_inner_size = self.CELL_SIZE - padding * 2
        x = col * self.CELL_SIZE + padding + cell_inner_size // 2
        y = vertical_offset + row * self.CELL_SIZE + padding + cell_inner_size // 2
        return x, y

    def _validate_render_inputs(self, grid, state, rescue_target_state):
        """Validate render inputs before drawing a frame.

        Args:
            grid (np.ndarray): Grid to draw.
            state (tuple[int, int]): Current drone position.
            rescue_target_state (dict): Rescue status by target coordinate.

        Raises:
            ValueError: If the grid shape or drone position is invalid.
            TypeError: If rescue target status is not dictionary-like.
        """
        if grid.shape != (self.rows, self.cols):
            raise ValueError(
                f"Grid shape must be {(self.rows, self.cols)}, received {grid.shape}."
            )

        row, col = state
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Drone state {state} is outside the rendered grid.")

        if not isinstance(rescue_target_state, dict):
            raise TypeError("rescue_target_state must be a dictionary.")

    def save_video(self):
        """Save the recorded frames to an MP4 video file.

        Video export is skipped when no frames were captured. This keeps cleanup
        safe for runs that failed before the first render call.

        Returns:
            None
        """
        if not self.frames:
            print("\nNo rendered frames available; skipping video save.")
            return

        os.makedirs(self.log_dir, exist_ok=True)
        video_path = os.path.join(self.log_dir, "simulation.mp4")

        try:
            imageio.mimsave(video_path, self.frames, fps=self.fps, macro_block_size=1)
        except Exception as error:
            raise RuntimeError(
                f"Failed to save simulation video to {video_path}: {error}"
            ) from error

        print(f"\nVideo saved to: {video_path}")

    def close(self):
        """Close the Pygame renderer and release display resources."""
        pygame.quit()
