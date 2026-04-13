import gymnasium as gym
import numpy as np
from gymnasium import spaces

class EnhancedFrozenLake(gym.Env):
    """
    Enhanced Frozen Lake POMDP Environment.
    - 8x8 grid (Option A).
    - 3x3 local observation window.
    - Shaped Rewards: Manhattan distance + Boundary penalty.
    - Non-stationary wind biasing movement every 100 episodes.
    """
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        self.size = 8 
        self.render_mode = render_mode
        
        # Mapping: 0: Frozen, 1: Start, 2: Goal, 3: Hole, 4: Wall
        self.TILE_TYPES = {'F': 0, 'S': 1, 'G': 2, 'H': 3, 'W': 4}
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=4, shape=(3, 3), dtype=np.int32)
        
        self.grid = self._generate_grid()
        self.agent_pos = None
        self.episode_counter = 0
        self.wind_direction = 0 
        self.wind_probability = 0.2 

    def _generate_grid(self):
        grid = np.full((self.size, self.size), self.TILE_TYPES['F'], dtype=np.int32)
        grid[0, 0] = self.TILE_TYPES['S']
        grid[self.size-1, self.size-1] = self.TILE_TYPES['G']
        
        hole_positions = [(1,1), (1,3), (2,5), (3,0), (4,4), (5,1), (6,6), (1,6), (5,7)]
        for r, c in hole_positions:
            grid[r, c] = self.TILE_TYPES['H']
        return grid

    def _get_dist(self, pos):
        return (self.size - 1 - pos[0]) + (self.size - 1 - pos[1])

    def _get_observation(self):
        r, c = self.agent_pos
        obs = np.full((3, 3), self.TILE_TYPES['W'], dtype=np.int32)
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    obs[dr + 1, dc + 1] = self.grid[nr, nc]
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Shift wind every 1000 episodes for stability
        self.wind_direction = (self.episode_counter // 1000) % 4
        self.episode_counter += 1
        self.agent_pos = (0, 0)
        return self._get_observation(), {"episode": self.episode_counter, "wind": self.wind_direction}

    def step(self, action):
        moves = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
        actual_action = action
        if self.np_random.random() < self.wind_probability:
            actual_action = self.wind_direction
            
        dr, dc = moves[actual_action]
        r, c = self.agent_pos
        dist_before = self._get_dist((r, c))
        
        nr, nc = r + dr, c + dc
        reward = -0.01 
        
        if 0 <= nr < self.size and 0 <= nc < self.size:
            self.agent_pos = (nr, nc)
            dist_after = self._get_dist((nr, nc))
            if dist_after < dist_before:
                reward += 0.1 # Amplified Progress bonus
            elif dist_after > dist_before:
                reward -= 0.1 # Amplified Regression penalty
        else:
            reward -= 0.2 # Boundary penalty
        
        tile = self.grid[self.agent_pos]
        terminated = False
        if tile == self.TILE_TYPES['G']:
            reward += 5.0 # Massive Goal reward
            terminated = True
        elif tile == self.TILE_TYPES['H']:
            reward -= 1.0
            terminated = True
            
        return self._get_observation(), reward, terminated, False, {}

    def render(self):
        if self.render_mode == "ansi":
            res = ""
            for r in range(self.size):
                for c in range(self.size):
                    if (r, c) == self.agent_pos: res += "A"
                    else:
                        tile = self.grid[r, c]
                        res += [k for k, v in self.TILE_TYPES.items() if v == tile][0]
                res += "\n"
            return res

if __name__ == "__main__":
    env = EnhancedFrozenLake()
    obs, info = env.reset()
    print(f"Reset Obs:\n{obs}")
