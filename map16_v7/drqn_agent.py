import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from datetime import datetime
from enhanced_frozen_lake import EnhancedFrozenLake

# --- 1. Multi-Input DRQN (256 Hidden Units) ---
class DRQN(nn.Module):
    def __init__(self, n_actions=4, embedding_dim=16, hidden_dim=256):
        super(DRQN, self).__init__()
        self.vision_embedding = nn.Embedding(5, embedding_dim)
        self.wind_embedding = nn.Embedding(4, embedding_dim)
        self.feature_layer = nn.Sequential(
            nn.Linear((49 + 1) * embedding_dim, hidden_dim),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_q = nn.Linear(hidden_dim, n_actions)

    def forward(self, vision, wind, hidden_state=None):
        b, s, h, w = vision.size()
        v_feat = self.vision_embedding(vision.long()).view(b, s, -1)
        w_feat = self.wind_embedding(wind.long())
        combined = torch.cat([v_feat, w_feat], dim=2)
        x = self.feature_layer(combined)
        lstm_out, new_hidden = self.lstm(x, hidden_state)
        q_values = self.fc_q(lstm_out)
        return q_values, new_hidden

    def init_hidden(self, batch_size=1, device="cpu"):
        return (torch.zeros(1, batch_size, 256).to(device),
                torch.zeros(1, batch_size, 256).to(device))

# --- 2. Episode-based Replay Buffer ---
class EpisodeReplayBuffer:
    def __init__(self, capacity=20000, sequence_length=20): # REDUCED to 20 for LSTM Stability
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
                o = list(o) + [np.zeros((7,7))]*pad
                w = list(w) + [0]*pad
                a = list(a) + [0]*pad
                r = list(r) + [0.0]*pad
                no = list(no) + [np.zeros((7,7))]*pad
                nw = list(nw) + [0]*pad
                d = list(d) + [True]*pad
            b_obs.append(o); b_wind.append(w); b_act.append(a); b_rew.append(r)
            b_next_obs.append(no); b_next_wind.append(nw); b_done.append(d)
        return (torch.FloatTensor(np.array(b_obs)), torch.LongTensor(np.array(b_wind)),
                torch.LongTensor(np.array(b_act)), torch.FloatTensor(np.array(b_rew)),
                torch.FloatTensor(np.array(b_next_obs)), torch.LongTensor(np.array(b_next_wind)),
                torch.BoolTensor(np.array(b_done)))

def evaluate(model, device, num_holes, wind_prob, num_episodes=20):
    """Run completely GREEDY evaluation (epsilon = 0.0) to find the True Success Rate."""
    env = EnhancedFrozenLake()
    env.num_holes = num_holes
    env.wind_probability = wind_prob
    success_count = 0
    
    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep+1000) # Fixed evaluation seeds for consistency
        wind = info['wind']
        hidden = model.init_hidden(1, device=device)
        done = False
        
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(device)
            wind_t = torch.LongTensor([wind]).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values, hidden = model(obs_t, wind_t, hidden)
            action = q_values[0, -1].argmax().item()
            obs, reward, terminated, truncated, info = env.step(action)
            wind = info.get('wind', wind)
            if terminated and reward > 20: success_count += 1
            done = terminated or truncated
            
    return (success_count / num_episodes) * 100

# --- 3. Training Loop with CONTINUOUS Curriculum ---
def train():
    env = EnhancedFrozenLake()
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    
    # Curriculum Start State
    env.num_holes = 10
    env.wind_probability = 0.0
    
    print(f"Training on device: {device} | CONTINUOUS CURRICULUM MODE")
    
    BATCH_SIZE = 64
    GAMMA = 0.995 
    LR = 1e-4
    EPS_DECAY = 0.9998 # Faster decay to allow exploitation
    MAX_EPISODES = 100000 
    
    policy_net = DRQN().to(device)
    target_net = DRQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    buffer = EpisodeReplayBuffer()
    epsilon = 1.0
    
    reward_history = deque(maxlen=100)

    for ep in range(1, MAX_EPISODES + 1):
        obs, info = env.reset()
        wind = info['wind']
        hidden = policy_net.init_hidden(1, device=device)
        episode_data = []
        total_reward = 0
        
        for t in range(400):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(device)
            wind_t = torch.LongTensor([wind]).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values, next_hidden = policy_net(obs_t, wind_t, hidden)
            action = env.action_space.sample() if random.random() < epsilon else q_values[0, -1].argmax().item()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_data.append((obs, wind, action, reward, next_obs, wind, done))
            obs, hidden, total_reward = next_obs, next_hidden, total_reward + reward
            if done: break
        
        buffer.push(episode_data)
        reward_history.append(total_reward)
        epsilon = max(0.01, epsilon * EPS_DECAY)

        if len(buffer.memory) > BATCH_SIZE:
            b_obs, b_wind, b_act, b_rew, b_nobs, b_nwind, b_done = [x.to(device) for x in buffer.sample(BATCH_SIZE)]
            q_out, _ = policy_net(b_obs, b_wind, policy_net.init_hidden(BATCH_SIZE, device))
            q_values = q_out.gather(2, b_act.unsqueeze(2)).squeeze(2)
            with torch.no_grad():
                targets = b_rew + (GAMMA * target_net(b_nobs, b_nwind, target_net.init_hidden(BATCH_SIZE, device))[0].max(dim=2)[0] * (~b_done))
            loss = F.smooth_l1_loss(q_values, targets)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
            optimizer.step()

        if ep % 20 == 0: target_net.load_state_dict(policy_net.state_dict())
        
        # CONTINUOUS CURRICULUM LOGIC
        if ep % 500 == 0:
            # We must use TRUE Evaluation Success Rate, without Epsilon randomness!
            eval_sr = evaluate(policy_net, device, env.num_holes, env.wind_probability)
            
            if eval_sr >= 40.0 and env.num_holes < 40:
                env.num_holes += 2
                print(f"\n>>> LEVEL UP: Now using {env.num_holes} holes! (True Eval SR: {eval_sr:.1f}%)")
            if eval_sr >= 40.0 and env.num_holes >= 25 and env.wind_probability == 0.0:
                env.wind_probability = 0.1
                print(f"\n>>> LEVEL UP: Wind enabled (0.1)! (True Eval SR: {eval_sr:.1f}%)")

            print(f"Ep: {ep} | Holes: {env.num_holes} | Wind: {'ON' if env.wind_probability > 0 else 'OFF'} | Train Avg Rew: {np.mean(reward_history):.2f} | Eval SR: {eval_sr:.1f}% | Eps: {epsilon:.3f}")

    torch.save(policy_net.state_dict(), f"map16_v7_curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")

if __name__ == "__main__":
    train()
