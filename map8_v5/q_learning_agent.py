import numpy as np
import random
import pickle
import os
from collections import defaultdict
from datetime import datetime
from enhanced_frozen_lake import EnhancedFrozenLake

class QLearningAgent:
    def __init__(self, n_actions=4, learning_rate=0.1, gamma=0.99):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        # Use a defaultdict for the Q-table so unseen states start at 0
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def _state_to_key(self, obs, wind):
        # Convert the 5x5 numpy array observation and wind into a hashable tuple
        return tuple(obs.flatten()) + (wind,)

    def get_action(self, obs, wind, epsilon=0.0):
        state_key = self._state_to_key(obs, wind)
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            # Argmax with random tie-breaking
            q_values = self.q_table[state_key]
            max_q = np.max(q_values)
            best_actions = [a for a in range(self.n_actions) if q_values[a] == max_q]
            return random.choice(best_actions)

    def update(self, obs, wind, action, reward, next_obs, next_wind, done):
        state_key = self._state_to_key(obs, wind)
        next_state_key = self._state_to_key(next_obs, next_wind)
        
        best_next_q = 0 if done else np.max(self.q_table[next_state_key])
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[state_key][action]
        
        self.q_table[state_key][action] += self.lr * td_error

    def save(self, filename):
        # Convert defaultdict to normal dict for pickling
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            q_dict = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(self.n_actions), q_dict)


def train_q_learning():
    env = EnhancedFrozenLake()
    agent = QLearningAgent()
    
    print("Training Tabular Q-Learning Baseline on map8_v5...")
    
    MAX_EPISODES = 150000 
    MAX_STEPS = 200
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY = 0.99995
    epsilon = EPS_START
    
    reward_history = deque(maxlen=100)
    success_history = deque(maxlen=100)

    for ep in range(1, MAX_EPISODES + 1):
        obs, info = env.reset()
        wind = info['wind']
        total_reward = 0
        reached_goal = False
        
        for t in range(MAX_STEPS):
            action = agent.get_action(obs, wind, epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_wind = info.get('wind', wind)
            
            if terminated and reward > 10:
                reached_goal = True
                
            done = terminated or truncated
            agent.update(obs, wind, action, reward, next_obs, next_wind, done)
            
            obs = next_obs
            wind = next_wind
            total_reward += reward
            if done: break
            
        reward_history.append(total_reward)
        success_history.append(1.0 if reached_goal else 0.0)
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        if ep % 1000 == 0:
            avg_rew = np.mean(reward_history)
            sr = np.mean(success_history) * 100
            print(f"Ep: {ep} | Q-Table Size: {len(agent.q_table)} states | Avg Rew: {avg_rew:.2f} | SR: {sr:.1f}% | Eps: {epsilon:.3f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"map8_v5_Q-Learning_{ts}.pkl"
    agent.save(filename)
    print(f"Q-Learning Model saved: {filename}")

if __name__ == "__main__":
    from collections import deque
    train_q_learning()
