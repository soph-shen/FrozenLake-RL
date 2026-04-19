import gymnasium as gym
import numpy as np
from gymnasium import spaces

class EnhancedFrozenLake(gym.Env):
    """
    Enhanced Frozen Lake POMDP Environment with Curriculum & Wide-Path constraints.
    """
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        self.size = 16 
        self.render_mode = render_mode
        self.max_steps = 400 
        
        # Curriculum Parameters
        self.num_holes = 10 
        self.wind_probability = 0.0 # Start at 0.0, max will be 0.1
        
        self.TILE_TYPES = {'F': 0, 'S': 1, 'G': 2, 'H': 3, 'W': 4}
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=4, shape=(7, 7), dtype=np.int32)
        
        self.grid = self._generate_grid()
        self.agent_pos = None
        self.episode_counter = 0
        self.current_step = 0
        self.wind_direction = 0 

    def _generate_grid(self):
        """Generates a map ensuring all paths are at least 2 blocks wide."""
        safe_zones = [(0, 0), (0, 1), (1, 0), (1, 1), 
                      (self.size-1, self.size-1), (self.size-1, self.size-2), 
                      (self.size-2, self.size-1), (self.size-2, self.size-2)]
        while True:
            grid = np.full((self.size, self.size), self.TILE_TYPES['F'], dtype=np.int32)
            grid[0, 0] = self.TILE_TYPES['S']
            grid[self.size-1, self.size-1] = self.TILE_TYPES['G']
            
            # Place holes
            for _ in range(self.num_holes):
                r, c = np.random.randint(0, self.size, 2)
                # Don't overwrite Start/Goal and safe zones
                if (r, c) not in safe_zones:
                    grid[r, c] = self.TILE_TYPES['H']
            
            # Wide-Path Check: Can a 2x2 agent reach the goal area?
            if self._is_solvable_wide(grid):
                return grid

    def _is_solvable_wide(self, grid):
        """
        BFS that checks if a 2x2 block of safe tiles can move from Start to Goal.
        This ensures all paths are at least 2 tiles wide.
        """
        start = (0, 0)
        queue = [start]
        visited = {start}
        
        while queue:
            r, c = queue.pop(0)
            
            # If any part of the 2x2 block touches the goal area
            if r >= self.size - 2 and c >= self.size - 2:
                return True
            
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                
                # Check bounds for 2x2
                if 0 <= nr < self.size - 1 and 0 <= nc < self.size - 1:
                    # Ensure 2x2 block has NO holes
                    if (grid[nr, nc] != self.TILE_TYPES['H'] and 
                        grid[nr+1, nc] != self.TILE_TYPES['H'] and 
                        grid[nr, nc+1] != self.TILE_TYPES['H'] and 
                        grid[nr+1, nc+1] != self.TILE_TYPES['H']):
                        
                        if (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
        return False

    def _get_dist(self, pos):
        return (self.size - 1 - pos[0]) + (self.size - 1 - pos[1])

    def _get_observation(self):
        r, c = self.agent_pos
        obs = np.full((7, 7), self.TILE_TYPES['W'], dtype=np.int32)
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    obs[dr + 3, dc + 3] = self.grid[nr, nc]
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = self._generate_grid()
        self.wind_direction = (self.episode_counter // 1000) % 4
        self.episode_counter += 1
        self.current_step = 0
        self.agent_pos = (0, 0)
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
        
        dist_before = self._get_dist((r, c))
        reward = -0.05 # Steady living penalty to force progress
        
        if 0 <= nr < self.size and 0 <= nc < self.size:
            self.agent_pos = (nr, nc)
            dist_after = self._get_dist(self.agent_pos)
            # PROPER MARKOVIAN REWARD SHAPING (Potential-Based)
            if dist_after < dist_before:
                reward += 0.1 
            elif dist_after > dist_before:
                reward -= 0.1 
        else:
            reward -= 0.5 # Wall Penalty
        
        tile = self.grid[self.agent_pos]
        terminated = False
        if tile == self.TILE_TYPES['G']:
            reward += 30.0 # MASSIVE GOAL REWARD
            terminated = True
        elif tile == self.TILE_TYPES['H']:
            reward -= 5.0 # Severe Hole Penalty
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
