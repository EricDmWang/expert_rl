import random
import os
from typing import Tuple, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces
try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:
    _PIL_OK = False


class GridWorld25v0(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, seed: int = 73, num_agents: int = 4, grid_size: int = 25,
                 max_steps: int = 200, gamma: float = 0.95, render_mode: str | None = None,
                 mode: str = "mode_2"):
        super().__init__()
        assert num_agents == 4, "This env is fixed to 4 agents as specified."
        assert mode in ["mode_1", "mode_2"], "Mode must be either 'mode_1' or 'mode_2'"
        
        self.grid_size = int(grid_size)
        self.num_agents = int(num_agents)
        # Fixed food count and level per spec
        self.num_food = 4
        self.food_level = 1
        # Episode control and discounting for reward shaping
        self.max_steps = int(max_steps)
        self.gamma = float(gamma)
        self.render_mode = render_mode
        self.mode = mode

        self.rng = np.random.default_rng(seed)
        
        # For mode_2: store fixed initial state
        self.fixed_agent_positions = None
        self.fixed_food_positions = None

        # Spaces: each agent chooses an action from {0..5} similar to LBF (noop, up, down, left, right, collect)
        self.action_space = spaces.Discrete(6)
        # Observation: 18-dim float32 per agent (stacked externally); here we expose per-agent obs builder
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)

        # State
        self.agent_positions: List[Tuple[int,int]] = []
        self.agent_levels: List[int] = []
        self.food_positions: List[Tuple[int,int,int]] = []  # (x,y,level)
        self.t = 0  # step counter
        # render output management
        self._render_out_dir = None
        self._render_frame_idx = 0
        self._render_frames = []  # list of saved frame file paths
        # agent path tracking for rendering
        self.agent_paths: List[List[Tuple[int,int]]] = [[] for _ in range(self.num_agents)]

    def seed(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def _sample_empty_cell(self) -> Tuple[int,int]:
        while True:
            x = int(self.rng.integers(0, self.grid_size))
            y = int(self.rng.integers(0, self.grid_size))
            if (x, y) not in self.agent_positions and all((x, y) != (fx, fy) for fx, fy, _ in self.food_positions):
                return x, y

    def _reset_state(self):
        self.agent_positions = []
        self.agent_levels = [1 for _ in range(self.num_agents)]
        self.food_positions = []
        self.t = 0
        # Reset agent paths
        self.agent_paths = [[] for _ in range(self.num_agents)]

        if self.mode == "mode_1":
            # Random initial conditions (current behavior)
            # Place agents
            for _ in range(self.num_agents):
                self.agent_positions.append(self._sample_empty_cell())

            # Place exactly 4 foods with fixed level = 1
            for _ in range(self.num_food):
                fx, fy = self._sample_empty_cell()
                self.food_positions.append((fx, fy, self.food_level))
                
        elif self.mode == "mode_2":
            # Fixed initial conditions
            if self.fixed_agent_positions is None or self.fixed_food_positions is None:
                # First time setup: generate and store fixed initial state
                # Place agents
                for _ in range(self.num_agents):
                    self.agent_positions.append(self._sample_empty_cell())

                # Place exactly 4 foods with fixed level = 1
                for _ in range(self.num_food):
                    fx, fy = self._sample_empty_cell()
                    self.food_positions.append((fx, fy, self.food_level))
                
                # Store the fixed initial state
                self.fixed_agent_positions = list(self.agent_positions)
                self.fixed_food_positions = list(self.food_positions)
            else:
                # Reset to stored fixed initial state
                self.agent_positions = list(self.fixed_agent_positions)
                self.food_positions = list(self.fixed_food_positions)

    def _neighbors_for_agent(self, idx: int) -> List[int]:
        # Fixed sharing pairs regardless of distance
        # 0-based agents: 0 gets (1,2), 1 gets (0,3), 2 gets (0,3), 3 gets (1,2)
        mapping = {0: [1,2], 1: [0,3], 2: [0,3], 3: [1,2]}
        return mapping[idx]

    def _obs_for_agent(self, idx: int) -> np.ndarray:
        # 18-dim vector: [sx,sy,sL, nx,ny,nL, f1x,f1y,f1L, f2x,f2y,f2L, f3x,f3y,f3L, f4x,f4y,f4L]
        sx, sy = self.agent_positions[idx]
        sL = self.agent_levels[idx]

        # neighbor info: two designated agents regardless of distance
        n_idxs = self._neighbors_for_agent(idx)
        nx, ny, nL = 0, 0, 0
        if len(n_idxs) >= 1:
            a = n_idxs[0]
            nx, ny = self.agent_positions[a]
            nL = self.agent_levels[a]

        # Food features: agent-specific visibility of exactly 3 out of 4 foods
        # Mapping by agent index (0-based):
        # 0 -> foods 0,1,2 ; 1 -> foods 1,2,3 ; 2 -> foods 2,3,0 ; 3 -> foods 3,0,1
        mapping = {
            0: [0, 1, 2],
            1: [1, 2, 3],
            2: [2, 3, 0],
            3: [3, 0, 1],
        }
        desired = mapping.get(idx, [0, 1, 2])
        m = len(self.food_positions)

        ffeat = []
        used = set()
        # add exactly three foods if available; zero-fill otherwise
        for k in range(3):
            if m > 0 and k < len(desired):
                sel = desired[k] % m
                # avoid duplicates when foods < 3
                tries = 0
                while sel in used and tries < m:
                    sel = (sel + 1) % m
                    tries += 1
                if sel not in used and m > 0:
                    fx, fy, fL = self.food_positions[sel]
                    ffeat += [fx, fy, fL]
                    used.add(sel)
                else:
                    ffeat += [0, 0, 0]
            else:
                ffeat += [0, 0, 0]
        # The 4th food slot is hidden -> zeros
        ffeat += [0, 0, 0]

        vec = [sx, sy, sL, nx, ny, nL] + ffeat
        return np.asarray(vec, dtype=np.float32)

    def _all_obs(self) -> np.ndarray:
        return np.stack([self._obs_for_agent(i) for i in range(self.num_agents)], axis=0)

    def setup(self, *, seed: int | None = None):
        """Setup the environment with initial state.
        
        For mode_1: Generates new random initial conditions
        For mode_2: Generates fixed initial conditions (first time) or resets to fixed state
        """
        if seed is not None:
            self.seed(seed)
        self._reset_state()
        return self._all_obs(), {}
    
    def reset(self, *, seed: int | None = None, options=None):
        """Reset the environment.
        
        For mode_1: Same as setup() - generates new random initial conditions
        For mode_2: Resets to the fixed initial state
        """
        if seed is not None:
            self.seed(seed)
        self._reset_state()
        obs = self._all_obs()
        info = {}
        return obs, info

    def step(self, actions: List[int]):
        # actions: list of ints of length num_agents
        assert isinstance(actions, (list, tuple)) and len(actions) == self.num_agents

        # Move agents (0: noop, 1: up, 2: down, 3: left, 4: right, 5: collect)
        def move(x, y, a):
            nx, ny = x, y
            if a == 1:   # up (decrease y)
                ny = y - 1
            elif a == 2: # down
                ny = y + 1
            elif a == 3: # left
                nx = x - 1
            elif a == 4: # right
                nx = x + 1
            # if move would go out of bounds, stay still
            if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                return x, y
            return nx, ny

        # Special move for collect action (5): move to adjacent food if possible
        def collect_move(x, y):
            # Check if adjacent to any food and move to it
            for fx, fy, _ in self.food_positions:
                if abs(x - fx) + abs(y - fy) == 1:  # adjacent
                    return fx, fy  # move to food position
            return x, y  # stay if no adjacent food

        # Apply movements with collision handling
        old_positions = list(self.agent_positions)
        proposed = []
        for i, a in enumerate(actions):
            # sanitize illegal action ids -> stay still (no-op)
            if not isinstance(a, (int, np.integer)) or a < 0 or a > 5:
                a = 0
            x, y = old_positions[i]
            if a in (1, 2, 3, 4):
                x, y = move(x, y, a)
            elif a == 5:  # collect action
                x, y = collect_move(x, y)
            proposed.append((x, y))

        # Rule 1: prevent two agents from swapping positions in a single step
        blocked = set()
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if proposed[i] == old_positions[j] and proposed[j] == old_positions[i]:
                    blocked.add(i)
                    blocked.add(j)

        # Rule 2: prevent multiple agents moving into the same target cell
        cell_to_agents = {}
        for i, pos in enumerate(proposed):
            cell_to_agents.setdefault(pos, []).append(i)
        for pos, inds in cell_to_agents.items():
            if len(inds) >= 2:
                for i in inds:
                    blocked.add(i)

        # Commit positions; blocked agents stay
        new_positions = []
        for i in range(self.num_agents):
            if i in blocked:
                new_positions.append(old_positions[i])
            else:
                new_positions.append(proposed[i])
                # Track agent path - add new position if it's different from previous
                if not self.agent_paths[i] or new_positions[i] != self.agent_paths[i][-1]:
                    self.agent_paths[i].append(new_positions[i])
        self.agent_positions = new_positions

        # Rewards and collection (agents collect food when they land on it)
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        collected_indices = []
        for i, (ax, ay) in enumerate(self.agent_positions):
            # Check if agent is now on top of any food
            for j, (fx, fy, fL) in enumerate(self.food_positions):
                if (ax, ay) == (fx, fy):  # agent landed on food
                    # Immediate reward equals food level (no discounting here)
                    rewards[i] += float(fL)
                    collected_indices.append(j)

        # Remove collected foods from the environment
        if collected_indices:
            # Create boolean mask: True for foods to keep, False for foods to remove
            mask = np.ones(len(self.food_positions), dtype=bool)
            mask[collected_indices] = False  # Mark collected foods for removal
            
            # Filter out collected foods using the mask
            self.food_positions = [f for k, f in enumerate(self.food_positions) if mask[k]]

        # time step advance
        self.t += 1

        terminated = len(self.food_positions) == 0
        truncated = False
        info = {}
        if (not terminated) and (self.t >= self.max_steps):
            truncated = True
            # Terminal penalty if foods remain: apply -0.5 * n to every agent (no discount)
            remaining = len(self.food_positions)
            if remaining > 0:
                penalty = float(-0.5 * remaining)
                rewards += penalty
                info["penalty"] = penalty

        obs = self._all_obs()
        return obs, rewards, terminated, truncated, info

    def render(self, mode: str | None = None):
        # ensure output directory exists for this execution
        if self._render_out_dir is None:
            base_dir = "/home/dongmingwang/project/Expert_RL/expert_rl/results"
            os.makedirs(base_dir, exist_ok=True)
            i = 0
            while True:
                out_dir = os.path.join(base_dir, f"execution_{i}")
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)
                    self._render_out_dir = out_dir
                    self._render_frame_idx = 0
                    break
                i += 1

        # Produce an rgb frame that mimics lbforaging style: grid with colored tiles
        if not _PIL_OK:
            # Fallback text render
            grid = np.full((self.grid_size, self.grid_size), fill_value='.')
            for (fx, fy, _fL) in self.food_positions:
                grid[fy, fx] = 'F'
            for i, (x, y) in enumerate(self.agent_positions):
                grid[y, x] = str(i)
            ascii_frame = "\n".join("".join(row) for row in grid)
            # save ascii frame
            try:
                if self._render_out_dir is not None:
                    fname = os.path.join(self._render_out_dir, f"frame_{self._render_frame_idx:06d}.txt")
                    with open(fname, "w") as f:
                        f.write(ascii_frame)
                    self._render_frame_idx += 1
            except Exception:
                pass
            print(ascii_frame)
            return None

        tile = 24
        pad = 1
        # Add extra space for axis labels and titles
        axis_width = 50  # Increased for Y-axis labels and title
        axis_height = 60  # Increased for X-axis labels and title
        W = self.grid_size * tile + axis_width
        H = self.grid_size * tile + axis_height
        img = Image.new('RGB', (W, H), (240, 240, 240))
        draw = ImageDraw.Draw(img)

        # draw grid
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                x0 = x * tile + axis_width
                y0 = y * tile + axis_height
                draw.rectangle([x0, y0, x0 + tile - 1, y0 + tile - 1], outline=(200, 200, 200), fill=(255, 255, 255))

        # helper to draw an apple icon
        def draw_apple(cx, cy, size):
            r = size // 2
            # apple body (red)
            draw.ellipse([cx - r, cy - r + 2, cx + r, cy + r], fill=(200, 40, 40), outline=(150, 20, 20))
            # stem (brown)
            stem_h = max(3, size // 4)
            draw.line([ (cx, cy - r + 2), (cx, cy - r - stem_h + 2) ], fill=(120, 70, 20), width=max(1, size // 12))
            # leaf (green)
            leaf_w = max(3, size // 4)
            leaf_h = max(2, size // 6)
            draw.ellipse([cx + 2, cy - r - leaf_h, cx + 2 + leaf_w, cy - r + leaf_h], fill=(60, 170, 60), outline=(40, 120, 40))

        # foods as apples
        for (fx, fy, _fL) in self.food_positions:
            cx = fx * tile + tile // 2 + axis_width
            cy = fy * tile + tile // 2 + axis_height
            apple_size = max(12, (tile * 2) // 3)
            draw_apple(cx, cy, apple_size)

        # communication links between designated pairs
        centers = []
        for (ax, ay) in self.agent_positions:
            cx = ax * tile + tile // 2 + axis_width
            cy = ay * tile + tile // 2 + axis_height
            centers.append((cx, cy))
        links = [(0,1), (0,2), (1,3), (2,3)]
        # helper to draw dashed line
        def draw_dashed_line(p0, p1, dash_len=8, gap_len=6, color=(150, 230, 150), width=2):
            x0, y0 = p0; x1, y1 = p1
            dx = x1 - x0; dy = y1 - y0
            dist = (dx*dx + dy*dy) ** 0.5
            if dist == 0:
                return
            ux, uy = dx / dist, dy / dist
            d = 0.0
            while d < dist:
                sx = int(x0 + ux * d)
                sy = int(y0 + uy * d)
                ex = int(x0 + ux * min(d + dash_len, dist))
                ey = int(y0 + uy * min(d + dash_len, dist))
                draw.line([(sx, sy), (ex, ey)], fill=color, width=width)
                d += dash_len + gap_len
        for a, b in links:
            if a < len(centers) and b < len(centers):
                draw_dashed_line(centers[a], centers[b], dash_len=max(6, tile//3), gap_len=max(4, tile//4), color=(160, 235, 160), width=2)

        # agents as small robot icons
        colors = [(60,120,216), (216,120,60), (160,60,200), (60,180,120)]
        def draw_robot(i, ax, ay):
            c = colors[i % len(colors)]
            x0 = ax * tile + pad + axis_width
            y0 = ay * tile + pad + axis_height
            bw = tile - 2 * pad
            bh = tile - 2 * pad
            # body
            draw.rectangle([x0, y0 + 4, x0 + bw, y0 + bh], fill=c, outline=(20,20,20))
            # head
            head_h = max(8, bh // 3)
            draw.rectangle([x0 + bw//6, y0, x0 + bw - bw//6, y0 + head_h], fill=c, outline=(20,20,20))
            # antenna
            draw.line([ (x0 + bw//2, y0), (x0 + bw//2, y0 - max(4, tile//6)) ], fill=(50,50,50), width=2)
            draw.ellipse([x0 + bw//2 - 2, y0 - max(4, tile//6) - 2, x0 + bw//2 + 2, y0 - max(4, tile//6) + 2], fill=(220,220,220))
            # eyes
            ey = y0 + head_h//2
            ex1 = x0 + bw//3
            ex2 = x0 + 2*bw//3
            eye_r = max(2, tile//12)
            draw.ellipse([ex1 - eye_r, ey - eye_r, ex1 + eye_r, ey + eye_r], fill=(255,255,255), outline=(0,0,0))
            draw.ellipse([ex2 - eye_r, ey - eye_r, ex2 + eye_r, ey + eye_r], fill=(255,255,255), outline=(0,0,0))
            # label index on body
            try:
                draw.text((x0 + 4, y0 + bh - 12), str(i), fill=(255,255,255))
            except Exception:
                pass

        for i, (ax, ay) in enumerate(self.agent_positions):
            draw_robot(i, ax, ay)

        # Draw agent paths with dashed lines in agent colors
        colors = [(60,120,216), (216,120,60), (160,60,200), (60,180,120)]
        for i, path in enumerate(self.agent_paths):
            if len(path) > 1:
                agent_color = colors[i % len(colors)]
                # Draw dashed line connecting all positions in the path
                for j in range(1, len(path)):
                    prev_pos = path[j-1]
                    curr_pos = path[j]
                    # Convert grid coordinates to pixel coordinates
                    x0 = prev_pos[0] * tile + tile // 2 + axis_width
                    y0 = prev_pos[1] * tile + tile // 2 + axis_height
                    x1 = curr_pos[0] * tile + tile // 2 + axis_width
                    y1 = curr_pos[1] * tile + tile // 2 + axis_height
                    # Draw dashed line in agent color
                    draw_dashed_line((x0, y0), (x1, y1), dash_len=max(4, tile//6), gap_len=max(3, tile//8), 
                                   color=agent_color, width=2)

        # Draw axis labels and grid coordinates
        try:
            # X-axis labels (top) - positioned above the grid
            for x in range(0, self.grid_size, max(1, self.grid_size // 10)):  # Show every 10th or all if small
                x_pos = x * tile + tile // 2 + axis_width
                y_pos = 35  # Moved lower for better visibility
                # Draw black text with white outline for better visibility
                text = str(x)
                draw.text((x_pos - 5, y_pos), text, fill=(0, 0, 0), stroke_width=2, stroke_fill=(255, 255, 255))
            
            # Y-axis labels (left) - positioned properly to the left of the grid
            for y in range(0, self.grid_size, max(1, self.grid_size // 10)):  # Show every 10th or all if small
                x_pos = 15  # Moved further left to avoid overlap
                y_pos = y * tile + tile // 2 + axis_height - 5
                # Draw black text with white outline for better visibility
                text = str(y)
                draw.text((x_pos, y_pos), text, fill=(0, 0, 0), stroke_width=2, stroke_fill=(255, 255, 255))
            
            # Axis titles - positioned clearly outside the grid area
            # X-axis title (top center, above X-axis labels)
            x_title_x = self.grid_size * tile // 2 + axis_width
            x_title_y = 20  # Positioned between top and labels
            draw.text((x_title_x - 5, x_title_y), "X", fill=(0, 0, 0), stroke_width=3, stroke_fill=(255, 255, 255))
            
            # Y-axis title (left center, to the left of Y-axis labels)
            y_title_x = 5  # Far left
            y_title_y = self.grid_size * tile // 2 + axis_height - 10
            draw.text((y_title_x, y_title_y), "Y", fill=(0, 0, 0), stroke_width=3, stroke_fill=(255, 255, 255))
            
        except Exception:
            pass  # If font rendering fails, continue without labels

        # Draw object position labels
        try:
            # Agent position labels - black text with white outline for clarity
            for i, (ax, ay) in enumerate(self.agent_positions):
                label_x = ax * tile + axis_width + 2
                label_y = ay * tile + axis_height - 18  # Moved slightly higher
                label_text = f"A{i}({ax},{ay})"
                # Draw black text with thick white outline for maximum visibility
                draw.text((label_x, label_y), label_text, fill=(0, 0, 0), 
                         stroke_width=3, stroke_fill=(255, 255, 255))
            
            # Food position labels - black text with white outline for clarity
            for i, (fx, fy, fL) in enumerate(self.food_positions):
                label_x = fx * tile + axis_width + 2
                label_y = fy * tile + axis_height + tile + 5  # Moved slightly lower
                label_text = f"F{i}({fx},{fy})"
                # Draw black text with thick white outline for maximum visibility
                draw.text((label_x, label_y), label_text, fill=(0, 0, 0), 
                         stroke_width=3, stroke_fill=(255, 255, 255))
                
        except Exception:
            pass  # If font rendering fails, continue without position labels

        frame = np.asarray(img)
        # save frame to disk
        try:
            if self._render_out_dir is not None:
                fname = os.path.join(self._render_out_dir, f"frame_{self._render_frame_idx:06d}.png")
                img.save(fname)
                self._render_frame_idx += 1
                self._render_frames.append(fname)
        except Exception:
            pass

        if mode == "rgb_array":
            return frame
        # human mode: do not display; image already saved above
        return None

    def save_animation(self, output_path: str | None = None, fps: int = 4):
        if not _PIL_OK:
            return None
        if not self._render_frames:
            return None
        try:
            frames = [Image.open(p).convert('RGB') for p in self._render_frames]
            if not frames:
                return None
            duration = int(1000 / max(1, fps))
            out_path = output_path
            if out_path is None:
                if self._render_out_dir is None:
                    return None
                out_path = os.path.join(self._render_out_dir, "animation.gif")
            frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
            return out_path
        except Exception:
            return None

    def close(self):
        # attempt to create an animation when closing
        try:
            self.save_animation()
        except Exception:
            pass
        return super().close()


