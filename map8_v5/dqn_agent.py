import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from datetime import datetime
from enhanced_frozen_lake import EnhancedFrozenLake

# --- 1. Multi-Input DQN (Vision + Wind, No LSTM) ---
class DQN(nn.Module):
    def __init__(self, n_actions=4, embedding_dim=16, hidden_dim=128):
        super(DQN, self).__init__()
        self.vision_embedding = nn.Embedding(5, embedding_dim)
        self.wind_embedding = nn.Embedding(4, embedding_dim)
        
        # 5x5 window = 25 tiles + 1 wind feature
        self.feature_layer = nn.Sequential(
            nn.Linear((25 + 1) * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_q = nn.Linear(hidden_dim, n_actions)

    def forward(self, vision, wind, hidden_state=None):
        # We accept hidden_state for API compatibility with play/evaluate scripts, but ignore it.
        b, s, h, w = vision.size()
        v_feat = self.vision_embedding(vision.long()).view(b, s, -1)
        w_feat = self.wind_embedding(wind.long())
        combined = torch.cat([v_feat, w_feat], dim=2)
        x = self.feature_layer(combined)
        q_values = self.fc_q(x)
        return q_values, None

    def init_hidden(self, batch_size=1, device="cpu"):
        return None

# --- 2. Standard Replay Buffer (Wraps transitions in lists to fake seq_len=1) ---
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        b_obs, b_wind, b_act, b_rew, b_next_obs, b_next_wind, b_done = zip(*transitions)
        return (torch.FloatTensor(np.array(b_obs)), torch.LongTensor(np.array(b_wind)),
                torch.LongTensor(np.array(b_act)), torch.FloatTensor(np.array(b_rew)),
                torch.FloatTensor(np.array(b_next_obs)), torch.LongTensor(np.array(b_next_wind)),
                torch.BoolTensor(np.array(b_done)))

# --- 3. Training Loop ---
def train():
    env = EnhancedFrozenLake()
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    
    print(f"Training on device: {device} | map8_v5 Standard DQN Multi-Input")
    
    BATCH_SIZE = 64
    GAMMA = 0.99
    LR = 1e-4
    EPS_DECAY = 0.99995
    MAX_EPISODES = 150000 
    
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    buffer = ReplayBuffer(capacity=100000)
    epsilon = 1.0
    
    reward_history = deque(maxlen=100)
    success_history = deque(maxlen=100)

    for ep in range(1, MAX_EPISODES + 1):
        obs, info = env.reset()
        wind = info['wind']
        total_reward, reached_goal = 0, False
        
        for t in range(200):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(device)
            wind_t = torch.LongTensor([wind]).unsqueeze(0).to(device)
            
            with torch.no_grad():
                q_values, _ = policy_net(obs_t, wind_t)
            
            action = env.action_space.sample() if random.random() < epsilon else q_values[0, -1].argmax().item()
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_wind = info.get('wind', wind)
            
            if terminated and reward > 10: reached_goal = True
            done = terminated or truncated
            
            buffer.push(([obs], [wind], [action], [reward], [next_obs], [next_wind], [done]))
            
            obs, wind = next_obs, next_wind
            total_reward += reward
            if done: break
            
        reward_history.append(total_reward)
        success_history.append(1.0 if reached_goal else 0.0)
        epsilon = max(0.05, epsilon * EPS_DECAY)

        if len(buffer.memory) > BATCH_SIZE:
            b_obs, b_wind, b_act, b_rew, b_nobs, b_nwind, b_done = [x.to(device) for x in buffer.sample(BATCH_SIZE)]
            
            q_values, _ = policy_net(b_obs, b_wind)
            q_values = q_values.gather(2, b_act.unsqueeze(2)).squeeze(2)
            
            with torch.no_grad():
                next_q_out, _ = target_net(b_nobs, b_nwind)
                targets = b_rew + (GAMMA * next_q_out.max(dim=2)[0] * (~b_done))
                
            loss = F.smooth_l1_loss(q_values, targets)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
            optimizer.step()

        if ep % 20 == 0: target_net.load_state_dict(policy_net.state_dict())
        if ep % 1000 == 0:
            print(f"Ep: {ep} | Avg Rew: {np.mean(reward_history):.2f} | SR: {np.mean(success_history)*100:.1f}% | Eps: {epsilon:.3f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"map8_v5_DQN_{ts}.pth"
    torch.save(policy_net.state_dict(), filename)
    print(f"Model saved: {filename}")

if __name__ == "__main__":
    train()
