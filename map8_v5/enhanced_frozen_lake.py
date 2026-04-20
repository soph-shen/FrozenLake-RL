import gymnasium as gym
import numpy as np
from gymnasium import spaces

class EnhancedFrozenLake(gym.Env):
    """
    Enhanced Frozen Lake POMDP Environment.
    - 8x8 grid.
    - 5x5 local observation window.
    - HURRY-UP REWARDS: High living penalty + Discovery bonuses.
    """
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        self.size = 8 
        self.render_mode = render_mode
        self.max_steps = 200 
        
        self.TILE_TYPES = {'F': 0, 'S': 1, 'G': 2, 'H': 3, 'W': 4}
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=4, shape=(5, 5), dtype=np.int32)
        
        self.grid = self._generate_grid()
        self.agent_pos = None
        self.episode_counter = 0
        self.current_step = 0
        self.min_dist_this_ep = 999 # Track best progress
        self.wind_direction = 0 
        self.wind_probability = 0.05 # Decreased from 0.1

    def _generate_grid(self):
        safe_zones = [(0, 0), (0, 1), (1, 0), (1, 1), 
                      (self.size-1, self.size-1), (self.size-1, self.size-2), 
                      (self.size-2, self.size-1), (self.size-2, self.size-2)]
        while True:
            grid = np.full((self.size, self.size), self.TILE_TYPES['F'], dtype=np.int32)
            grid[0, 0] = self.TILE_TYPES['S']
            grid[self.size-1, self.size-1] = self.TILE_TYPES['G']
            for _ in range(12): 
                r, c = np.random.randint(0, self.size, 2)
                if (r, c) not in safe_zones:
                    grid[r, c] = self.TILE_TYPES['H']
            if self._is_solvable(grid): return grid

    def _is_solvable(self, grid):
        start, goal = (0, 0), (self.size - 1, self.size - 1)
        queue, visited = [start], {start}
        while queue:
            curr_r, curr_c = queue.pop(0)
            if (curr_r, curr_c) == goal: return True
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if grid[nr, nc] != self.TILE_TYPES['H'] and (nr, nc) not in visited:
                        visited.add((nr, nc)); queue.append((nr, nc))
        return False

    def _get_dist(self, pos):
        return (self.size - 1 - pos[0]) + (self.size - 1 - pos[1])

    def _get_observation(self):
        r, c = self.agent_pos
        obs = np.full((5, 5), self.TILE_TYPES['W'], dtype=np.int32)
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    obs[dr + 2, dc + 2] = self.grid[nr, nc]
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # IMPORTANT: Generate a NEW grid on every reset to prevent overfitting
        self.grid = self._generate_grid()
        
        self.wind_direction = (self.episode_counter // 1000) % 4
        self.episode_counter += 1
        self.current_step = 0
        self.agent_pos = (0, 0)
        self.min_dist_this_ep = self._get_dist(self.agent_pos)
        return self._get_observation(), {"episode": self.episode_counter, "wind": self.wind_direction}

    def step(self, action):
        self.current_step += 1
        moves = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
        actual_action = action
        if self.np_random.random() < self.wind_probability:
            actual_action = self.wind_direction
            
        dr, dc = moves[actual_action]
        r, c = self.agent_pos
        
        nr, nc = r + dr, c + dc
        reward = -0.1 # HIGH Living Penalty to force speed
        
        if 0 <= nr < self.size and 0 <= nc < self.size:
            self.agent_pos = (nr, nc)
            current_dist = self._get_dist(self.agent_pos)
            
            # Discovery Reward: Only reward NEW progress
            if current_dist < self.min_dist_this_ep:
                reward += 0.5 # Big bonus for getting closer than ever before
                self.min_dist_this_ep = current_dist
        else:
            reward -= 1.0 # Wall Penalty
        
        tile = self.grid[self.agent_pos]
        terminated = False
        if tile == self.TILE_TYPES['G']:
            reward += 15.0 # Massive Final Reward
            terminated = True
        elif tile == self.TILE_TYPES['H']:
            reward -= 2.0 # More severe Hole Penalty
            terminated = True
            
        truncated = self.current_step >= self.max_steps
            
        return self._get_observation(), reward, terminated, truncated, {}

    def render(self):
        res = ""
        for r in range(self.size):
            for c in range(self.size):
                if (r, c) == self.agent_pos: res += "A"
                else:
                    tile = self.grid[r, c]
                    res += [k for k, v in self.TILE_TYPES.items() if v == tile][0]
            res += "\n"
        return res
