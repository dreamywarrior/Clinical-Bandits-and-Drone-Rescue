"""Pygame renderer for the drone rescue environment."""

import imageio
import os

import numpy as np
import pygame


class GridRenderer:
    """Visual renderer for the drone rescue environment."""

    CELL_SIZE = 95

    WIDTH = 1296
    HEIGHT = 736

    BG_COLOR = (8, 12, 28)

    BOX_COLOR = (36, 38, 52)
    BORDER_COLOR = (98, 102, 145)

    TEXT_COLOR = (240, 240, 245)
    WHITE = (255, 255, 255)

    PATH_COLOR = (30, 245, 190)

    ARROW_COLOR = (16, 22, 38)

    VALUE_TEXT_COLOR = (20, 24, 36)

    LOW_VALUE_COLOR = (235, 75, 80)
    HIGH_VALUE_COLOR = (70, 245, 120)

    # =====================================================
    # GRID COLORS
    # =====================================================

    COLORS = {
        -1: (90, 90, 100),     # Blocked
        0: (238, 238, 238),   # Free
        1: (45, 200, 70),     # Start
        2: (255, 220, 40),    # Wind
        3: (235, 70, 70),     # Danger
        4: (40, 110, 255),    # Charging
        5: (170, 60, 255),    # Rescue
    }

    def __init__(
        self,
        rows=5,
        cols=5,
        fps=20,
        log_dir="logs",
    ):

        if rows <= 0 or cols <= 0:
            raise ValueError(
                "Renderer rows and columns must be positive integers."
            )

        pygame.init()

        self.screen = pygame.display.set_mode(
            (self.WIDTH, self.HEIGHT)
        )

        pygame.display.set_caption(
            "Drone Rescue Environment"
        )

        self.rows = rows
        self.cols = cols

        self.fps = fps
        self.log_dir = log_dir

        self.font = pygame.font.SysFont(
            "Segoe UI",
            22,
        )

        self.small_font = pygame.font.SysFont(
            "Segoe UI",
            18,
        )

        self.title_font = pygame.font.SysFont(
            "Segoe UI",
            40,
            bold=True,
        )

        self.heading_font = pygame.font.SysFont(
            "Segoe UI",
            30,
            bold=True,
        )

        # Smaller font to avoid overlap with arrows.
        self.symbol_font = pygame.font.SysFont(
            "Segoe UI",
            16,
            bold=True,
        )

        self.frames = []

        self.path_history = []

    # =====================================================
    # DRAW BOX
    # =====================================================

    def draw_box(
        self,
        x,
        y,
        width,
        height,
        radius=20,
    ):

        rect = pygame.Rect(
            x,
            y,
            width,
            height,
        )

        pygame.draw.rect(
            self.screen,
            self.BOX_COLOR,
            rect,
            border_radius=radius,
        )

        pygame.draw.rect(
            self.screen,
            self.BORDER_COLOR,
            rect,
            2,
            border_radius=radius,
        )

    # =====================================================
    # LEGEND ITEM
    # =====================================================

    def draw_legend_item(
        self,
        x,
        y,
        item,
        text,
    ):

        center_x = x
        center_y = y + 11

        # =================================================
        # PATH
        # =================================================

        if text == "Path":

            pygame.draw.line(
                self.screen,
                self.PATH_COLOR,
                (center_x - 10, center_y),
                (center_x + 10, center_y),
                width=5,
            )

        # =================================================
        # DRONE
        # =================================================

        elif item == "DRONE":

            pygame.draw.circle(
                self.screen,
                (0, 255, 255),
                (center_x, center_y),
                9,
            )

            pygame.draw.circle(
                self.screen,
                (200, 255, 255),
                (center_x, center_y),
                9,
                width=2,
            )

        # =================================================
        # SYMBOL ITEMS
        # =================================================

        elif isinstance(item, str):

            symbol_surface = self.symbol_font.render(
                item,
                True,
                self.WHITE,
            )

            symbol_rect = symbol_surface.get_rect(
                center=(center_x, center_y)
            )

            self.screen.blit(
                symbol_surface,
                symbol_rect,
            )

        legend_surface = self.small_font.render(
            text,
            True,
            (230, 230, 235),
        )

        self.screen.blit(
            legend_surface,
            (x + 28, y),
        )

    # =====================================================
    # MAIN RENDER
    # =====================================================

    def render(
        self,
        grid,
        state,
        battery_level,
        step_count,
        cumulative_reward,
        rescue_target_state,
        action_taken,
        value_function=None,
        policy=None,
    ):

        self.screen.fill(self.BG_COLOR)

        padding = 10

        grid_total_height = (
            self.rows * self.CELL_SIZE
        )

        vertical_offset = (
            (self.HEIGHT - grid_total_height) // 2
        ) + 20

        # =================================================
        # DRAW GRID
        # =================================================

        for row in range(self.rows):

            for col in range(self.cols):

                cell_value = grid[row, col]

                color = self.COLORS.get(
                    cell_value,
                    self.COLORS[0],
                )

                x = (
                    col * self.CELL_SIZE
                    + padding
                )

                y = (
                    vertical_offset
                    + row * self.CELL_SIZE
                    + padding
                )

                cell_size = (
                    self.CELL_SIZE
                    - (padding * 2)
                )

                rect = pygame.Rect(
                    x,
                    y,
                    cell_size,
                    cell_size,
                )

                pygame.draw.rect(
                    self.screen,
                    color,
                    rect,
                    border_radius=18,
                )

                # =========================================
                # VALUE FUNCTION
                # =========================================

                if (
                    value_function is not None
                    and value_function[row, col] >= 0
                ):

                    self._draw_value_heatmap_cell(
                        rect,
                        value_function[row, col],
                    )

                    self._draw_value_label(
                        rect,
                        value_function[row, col],
                    )

                pygame.draw.rect(
                    self.screen,
                    self.WHITE,
                    rect,
                    2,
                    border_radius=18,
                )

                # =========================================
                # SPECIAL SYMBOLS
                # =========================================

                self._draw_cell_symbol(
                    rect,
                    cell_value,
                )

        # =================================================
        # POLICY
        # =================================================

        if policy is not None:

            self._draw_policy_arrows(
                policy,
                vertical_offset,
                padding,
            )

        # =================================================
        # PATH
        # =================================================

        displayed_path = list(
            self.path_history
        )

        if state not in displayed_path:
            displayed_path.append(state)

        self._draw_behavior_path(
            displayed_path,
            vertical_offset,
            padding,
        )

        # =================================================
        # DRONE
        # =================================================

        drone_x, drone_y = self._cell_center(
            state,
            vertical_offset,
            padding,
        )

        pygame.draw.circle(
            self.screen,
            (0, 255, 255),
            (drone_x, drone_y),
            30,
        )

        # =================================================
        # STATUS PANEL
        # =================================================

        panel_x = (
            self.cols * self.CELL_SIZE
        ) + 45

        title_surface = self.title_font.render(
            "DRONE STATUS",
            True,
            self.WHITE,
        )

        self.screen.blit(
            title_surface,
            (panel_x, 36),
        )

        self.draw_box(
            panel_x,
            120,
            360,
            350,
        )

        policy_type = ("Algorithm" if policy is not None else "Fallback")

        status_lines = [
            ("State", state),
            ("Step", step_count),
            ("Battery", battery_level),
            (
                "Reward",
                round(cumulative_reward, 2),
            ),
            (
                "Action",
                action_taken or "None",
            ),
            ("Policy", policy_type),
        ]

        line_spacing = 58

        content_height = (
            len(status_lines)
            * line_spacing
        )

        start_y = (
            120
            + (350 - content_height) // 2
            + 10
        )

        label_x = panel_x + 48
        value_x = panel_x + 185

        # =================================================
        # BOLD LABEL FONT
        # =================================================

        bold_font = pygame.font.SysFont(
            "Segoe UI",
            22,
            bold=True,
        )

        for i, (label, value) in enumerate(
            status_lines
        ):

            y = start_y + i * line_spacing

            # Labels bold.
            label_surface = bold_font.render(
                f"{label}",
                True,
                self.TEXT_COLOR,
            )

            # Colon normal.
            colon_surface = self.font.render(
                ":",
                True,
                self.TEXT_COLOR,
            )

            # Values normal.
            value_surface = self.font.render(
                f"{value}",
                True,
                self.TEXT_COLOR,
            )

            self.screen.blit(
                label_surface,
                (label_x, y),
            )

            self.screen.blit(
                colon_surface,
                (label_x + 110, y),
            )

            self.screen.blit(
                value_surface,
                (value_x, y),
            )

        # =================================================
        # LEGEND PANEL
        # =================================================

        legend_x = panel_x + 385

        self.draw_box(
            legend_x,
            120,
            360,
            350,
        )

        legend_title = self.heading_font.render(
            "LEGEND",
            True,
            self.WHITE,
        )

        self.screen.blit(
            legend_title,
            (legend_x + 24, 145),
        )

        left_x = legend_x + 30
        right_x = legend_x + 205

        left_items = [
            ("S", "Start"),
            ("W", "Wind Zone"),
            ("X", "Blocked"),
            (self.PATH_COLOR, "Path"),
        ]

        right_items = [
            ("C", "Charging"),
            ("R", "Rescue"),
            ("D", "Danger Zone"),
            ("DRONE", "Drone"),
        ]

        start_y = 215

        spacing = 48

        for index, (item, text) in enumerate(
            left_items
        ):

            self.draw_legend_item(
                left_x,
                start_y + (index * spacing),
                item,
                text,
            )

        for index, (item, text) in enumerate(
            right_items
        ):

            self.draw_legend_item(
                right_x,
                start_y + (index * spacing),
                item,
                text,
            )

        policy_y = (
            start_y
            + ((len(right_items)) * spacing)
        )

        self.draw_arrow_legend_item(
            right_x,
            policy_y,
            "Policy",
        )

        # =================================================
        # RESCUE TARGETS PANEL
        # =================================================

        self.draw_box(
            panel_x,
            500,
            740,
            165,
        )

        rescue_title = self.heading_font.render(
            "RESCUE TARGETS",
            True,
            self.WHITE,
        )

        self.screen.blit(
            rescue_title,
            (panel_x + 24, 515),
        )

        for i, (target, active) in enumerate(
            rescue_target_state.items()
        ):

            target_surface = self.font.render(
                f"{target}",
                True,
                (240, 240, 240),
            )

            status = (
                "ACTIVE"
                if active
                else "DONE"
            )

            status_color = (
                (0, 220, 100)
                if active
                else (255, 140, 0)
            )

            status_surface = self.font.render(
                status,
                True,
                status_color,
            )

            y_pos = 570 + i * 48

            self.screen.blit(
                target_surface,
                (panel_x + 36, y_pos),
            )

            self.screen.blit(
                status_surface,
                (panel_x + 230, y_pos),
            )

        pygame.display.flip()

        # =================================================
        # SLOWER PLAYBACK
        # =================================================

        pygame.time.delay(1800)

        # =================================================
        # FRAME CAPTURE
        # =================================================

        frame = pygame.surfarray.array3d(
            self.screen
        )

        frame = np.transpose(
            frame,
            (1, 0, 2),
        )

        for _ in range(60):
            self.frames.append(frame.copy())

        if state not in self.path_history:
            self.path_history.append(state)

    # =====================================================
    # SPECIAL CELL SYMBOLS
    # =====================================================

    def _draw_cell_symbol(
        self,
        rect,
        cell_value,
    ):
        """Draw special-cell symbols."""

        symbol_map = {
            -1: "X",
            1: "S",
            2: "W",
            3: "D",
            4: "C",
            5: "R",
        }

        if cell_value not in symbol_map:
            return

        symbol = symbol_map[cell_value]

        symbol_surface = self.symbol_font.render(
            symbol,
            True,
            (20, 20, 20),
        )

        symbol_rect = symbol_surface.get_rect()

        # =================================================
        # BLOCKED CELL
        # =================================================

        if cell_value == -1:

            symbol_rect.center = rect.center

        # =================================================
        # OTHER SPECIAL CELLS
        # =================================================

        else:

            # Push symbols further right and lower
            # to avoid overlapping arrows.
            symbol_rect.bottomright = (
                rect.right - 6,
                rect.bottom - 4,
            )

        self.screen.blit(
            symbol_surface,
            symbol_rect,
        )

    # =====================================================
    # VALUE HEATMAP
    # =====================================================

    def _draw_value_heatmap_cell(
        self,
        rect,
        value,
    ):

        value = float(
            np.clip(value, 0.0, 1.0)
        )

        heat_color = self._value_to_color(
            value
        )

        overlay = pygame.Surface(
            (rect.width, rect.height),
            pygame.SRCALPHA,
        )

        pygame.draw.rect(
            overlay,
            (*heat_color, 145),
            overlay.get_rect(),
            border_radius=18,
        )

        self.screen.blit(
            overlay,
            rect.topleft,
        )

    def _draw_value_label(
        self,
        rect,
        value,
    ):

        label = f"{float(value):.2f}"

        value_surface = self.small_font.render(
            label,
            True,
            self.VALUE_TEXT_COLOR,
        )

        label_width = (
            value_surface.get_width()
            + 12
        )

        label_height = (
            value_surface.get_height()
            + 4
        )

        label_rect = pygame.Rect(
            0,
            0,
            label_width,
            label_height,
        )

        label_rect.centerx = rect.centerx

        label_rect.top = rect.top + 5

        background = pygame.Surface(
            (
                label_rect.width,
                label_rect.height,
            ),
            pygame.SRCALPHA,
        )

        pygame.draw.rect(
            background,
            (255, 255, 255, 210),
            background.get_rect(),
            border_radius=8,
        )

        self.screen.blit(
            background,
            label_rect.topleft,
        )

        self.screen.blit(
            value_surface,
            (
                label_rect.x + 6,
                label_rect.y + 2,
            ),
        )

    def _value_to_color(
        self,
        value,
    ):

        value = float(
            np.clip(value, 0.0, 1.0)
        )

        if value < 0.5:

            ratio = value / 0.5

            red = self.LOW_VALUE_COLOR[0]

            green = int(
                75 + ratio * 170
            )

        else:

            ratio = (
                (value - 0.5) / 0.5
            )

            red = int(
                235 - ratio * 165
            )

            green = self.HIGH_VALUE_COLOR[1]

        return red, green, 80

    # =====================================================
    # POLICY
    # =====================================================

    def _draw_policy_arrows(
        self,
        policy,
        vertical_offset,
        padding,
    ):

        for state, action in policy.items():

            center = self._cell_center(
                state,
                vertical_offset,
                padding,
            )

            adjusted_center = (
                center[0],
                center[1] + 12,
            )

            self._draw_policy_arrow(
                adjusted_center,
                action,
            )

    def _draw_policy_arrow(
        self,
        center,
        action,
    ):

        if action == 4:

            pygame.draw.circle(
                self.screen,
                self.ARROW_COLOR,
                center,
                9,
                width=3,
            )

            return

        direction_vectors = {
            0: (0, -1),
            1: (0, 1),
            2: (-1, 0),
            3: (1, 0),
        }

        dx, dy = direction_vectors[action]

        tail = (
            center[0] - dx * 10,
            center[1] - dy * 10,
        )

        tip = (
            center[0] + dx * 10,
            center[1] + dy * 10,
        )

        shaft_end = (
            tip[0] - dx * 6,
            tip[1] - dy * 6,
        )

        pygame.draw.line(
            self.screen,
            self.ARROW_COLOR,
            tail,
            shaft_end,
            width=5,
        )

        self._draw_arrow_head(
            tip,
            dx,
            dy,
        )

    def _draw_arrow_head(
        self,
        tip,
        dx,
        dy,
    ):

        perpendicular = (-dy, dx)

        left = (
            tip[0] - dx * 12
            + perpendicular[0] * 7,
            tip[1] - dy * 12
            + perpendicular[1] * 7,
        )

        right = (
            tip[0] - dx * 12
            - perpendicular[0] * 7,
            tip[1] - dy * 12
            - perpendicular[1] * 7,
        )

        pygame.draw.polygon(
            self.screen,
            self.ARROW_COLOR,
            [tip, left, right],
        )

    def draw_arrow_legend_item(
        self,
        x,
        y,
        text,
    ):

        center_y = y + 11

        start = (
            x - 10,
            center_y,
        )

        end = (
            x + 10,
            center_y,
        )

        pygame.draw.line(
            self.screen,
            self.WHITE,
            start,
            end,
            width=4,
        )

        pygame.draw.polygon(
            self.screen,
            self.WHITE,
            [
                (x + 10, center_y),
                (x + 1, center_y - 6),
                (x + 1, center_y + 6),
            ],
        )

        legend_surface = self.small_font.render(
            text,
            True,
            (230, 230, 235),
        )

        self.screen.blit(
            legend_surface,
            (x + 28, y),
        )

    # =====================================================
    # PATH
    # =====================================================

    def _draw_behavior_path(
        self,
        path,
        vertical_offset,
        padding,
    ):

        if not path:
            return

        points = [
            self._cell_center(
                state,
                vertical_offset,
                padding,
            )
            for state in path
        ]

        if len(points) > 1:

            pygame.draw.lines(
                self.screen,
                self.PATH_COLOR,
                False,
                points,
                width=6,
            )

        for point in points:

            pygame.draw.circle(
                self.screen,
                self.PATH_COLOR,
                point,
                4,
            )

    # =====================================================
    # CELL CENTER
    # =====================================================

    def _cell_center(
        self,
        state,
        vertical_offset,
        padding,
    ):

        row, col = state

        cell_inner_size = (
            self.CELL_SIZE
            - padding * 2
        )

        x = (
            col * self.CELL_SIZE
            + padding
            + cell_inner_size // 2
        )

        y = (
            vertical_offset
            + row * self.CELL_SIZE
            + padding
            + cell_inner_size // 2
        )

        return x, y

    # =====================================================
    # VIDEO EXPORT
    # =====================================================

    def save_video(self):

        if not self.frames:

            print(
                "\nNo rendered frames available; skipping video save."
            )

            return

        os.makedirs(
            self.log_dir,
            exist_ok=True,
        )

        video_path = os.path.join(
            self.log_dir,
            "simulation.mp4",
        )

        imageio.mimsave(
            video_path,
            self.frames,
            fps=self.fps,
            macro_block_size=1,
        )

        print(
            f"\nVideo saved to: {video_path}"
        )

    def close(self):

        pygame.quit()