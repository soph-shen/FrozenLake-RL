import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from datetime import datetime
from enhanced_frozen_lake import EnhancedFrozenLake

# --- 1. Multi-Input QR-DRQN (Vision + Wind) ---
class QR_DRQN(nn.Module):
    def __init__(self, n_actions=4, num_quantiles=51, embedding_dim=16, hidden_dim=128):
        super(QR_DRQN, self).__init__()
        self.num_quantiles = num_quantiles
        self.n_actions = n_actions
        self.vision_embedding = nn.Embedding(5, embedding_dim)
        self.wind_embedding = nn.Embedding(4, embedding_dim)
        
        # 5x5 window = 25 tiles + 1 wind feature
        self.feature_layer = nn.Sequential(
            nn.Linear((25 + 1) * embedding_dim, hidden_dim),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_q = nn.Linear(hidden_dim, n_actions * num_quantiles)

    def forward(self, vision, wind, hidden_state=None):
        b, s, h, w = vision.size()
        v_feat = self.vision_embedding(vision.long()).view(b, s, -1)
        w_feat = self.wind_embedding(wind.long())
        combined = torch.cat([v_feat, w_feat], dim=2)
        x = self.feature_layer(combined)
        lstm_out, new_hidden = self.lstm(x, hidden_state)
        
        quantiles = self.fc_q(lstm_out)
        quantiles = quantiles.view(b, s, self.n_actions, self.num_quantiles)
        return quantiles, new_hidden

    def init_hidden(self, batch_size=1, device="cpu"):
        return (torch.zeros(1, batch_size, 128).to(device),
                torch.zeros(1, batch_size, 128).to(device))

# --- 2. Episode-based Replay Buffer ---
class EpisodeReplayBuffer:
    def __init__(self, capacity=5000, sequence_length=20):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.memory = deque(maxlen=capacity)

    def push(self, episode):
        self.memory.append(episode)

    def sample(self, batch_size):
        sampled_episodes = random.sample(self.memory, batch_size)
        b_obs, b_wind, b_act, b_rew, b_next_obs, b_next_wind, b_done = [], [], [], [], [], [], []
        for ep in sampled_episodes:
            start = random.randint(0, max(0, len(ep) - self.sequence_length))
            seq = ep[start : start + self.sequence_length]
            o, w, a, r, no, nw, d = zip(*seq)
            if len(o) < self.sequence_length:
                pad = self.sequence_length - len(o)
                o = list(o) + [np.zeros((5,5))]*pad
                w = list(w) + [0]*pad
                a = list(a) + [0]*pad
                r = list(r) + [0.0]*pad
                no = list(no) + [np.zeros((5,5))]*pad
                nw = list(nw) + [0]*pad
                d = list(d) + [True]*pad
            b_obs.append(o); b_wind.append(w); b_act.append(a); b_rew.append(r)
            b_next_obs.append(no); b_next_wind.append(nw); b_done.append(d)
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
    
    print(f"Training on device: {device} | map8_v5 QR-DRQN Multi-Input")
    
    BATCH_SIZE = 32
    GAMMA = 0.99
    LR = 1e-4
    EPS_DECAY = 0.99995
    MAX_EPISODES = 150000 
    
    policy_net = QR_DRQN().to(device)
    target_net = QR_DRQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    buffer = EpisodeReplayBuffer()
    epsilon = 1.0
    
    reward_history = deque(maxlen=100)
    success_history = deque(maxlen=100)

    for ep in range(1, MAX_EPISODES + 1):
        obs, info = env.reset()
        wind = info['wind']
        hidden = policy_net.init_hidden(1, device=device)
        episode_data = []
        total_reward, reached_goal = 0, False
        
        for t in range(200):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(device)
            wind_t = torch.LongTensor([wind]).unsqueeze(0).to(device)
            with torch.no_grad():
                quantiles, next_hidden = policy_net(obs_t, wind_t, hidden)
                q_values = quantiles.mean(dim=3) # Average over quantiles to get Q-value
            action = env.action_space.sample() if random.random() < epsilon else q_values[0, -1].argmax().item()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            if terminated and reward > 10: reached_goal = True
            done = terminated or truncated
            episode_data.append((obs, wind, action, reward, next_obs, wind, done))
            obs, hidden, total_reward = next_obs, next_hidden, total_reward + reward
            if done: break
        
        buffer.push(episode_data)
        reward_history.append(total_reward)
        success_history.append(1.0 if reached_goal else 0.0)
        epsilon = max(0.05, epsilon * EPS_DECAY)

        if len(buffer.memory) > BATCH_SIZE:
            b_obs, b_wind, b_act, b_rew, b_nobs, b_nwind, b_done = [x.to(device) for x in buffer.sample(BATCH_SIZE)]
            
            # Get current quantiles
            quantiles, _ = policy_net(b_obs, b_wind, policy_net.init_hidden(BATCH_SIZE, device))
            
            # Select quantiles for the taken actions
            act_idx = b_act.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, policy_net.num_quantiles)
            current_quantiles = quantiles.gather(2, act_idx).squeeze(2) # [batch, seq, num_quantiles]
            
            with torch.no_grad():
                # Get next quantiles from target network
                next_quantiles, _ = target_net(b_nobs, b_nwind, target_net.init_hidden(BATCH_SIZE, device))
                next_q_values = next_quantiles.mean(dim=3)
                next_actions = next_q_values.argmax(dim=2) # [batch, seq]
                
                # Select quantiles for best next actions
                next_act_idx = next_actions.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, policy_net.num_quantiles)
                target_quantiles = next_quantiles.gather(2, next_act_idx).squeeze(2) # [batch, seq, num_quantiles]
                
                # Compute targets
                b_rew_exp = b_rew.unsqueeze(2).expand(-1, -1, policy_net.num_quantiles)
                b_done_exp = b_done.unsqueeze(2).expand(-1, -1, policy_net.num_quantiles)
                targets = b_rew_exp + (GAMMA * target_quantiles * (~b_done_exp))
            
            # Compute Quantile Huber Loss
            q = current_quantiles.unsqueeze(3) # [batch, seq, num_quantiles, 1]
            target_q = targets.unsqueeze(2) # [batch, seq, 1, num_quantiles]
            
            diff = target_q - q
            huber_loss = F.smooth_l1_loss(q, target_q, reduction='none')
            
            N = policy_net.num_quantiles
            tau = (torch.arange(N, device=device, dtype=torch.float32) + 0.5) / N
            tau = tau.view(1, 1, N, 1)
            
            weight = torch.abs(tau - (diff < 0).float())
            loss = (weight * huber_loss).mean(dim=2).sum(dim=2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
            optimizer.step()

        if ep % 10 == 0: target_net.load_state_dict(policy_net.state_dict())
        if ep % 100 == 0:
            print(f"Ep: {ep} | Avg Rew: {np.mean(reward_history):.2f} | SR: {np.mean(success_history)*100:.1f}% | Eps: {epsilon:.3f}")

    torch.save(policy_net.state_dict(), f"map8_v5_QR-DRQN_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")

if __name__ == "__main__":
    train()
