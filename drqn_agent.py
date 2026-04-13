import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from enhanced_frozen_lake import EnhancedFrozenLake

# --- 1. DRQN Model Architecture ---
class DRQN(nn.Module):
    def __init__(self, n_actions=4, embedding_dim=16, hidden_dim=128):
        super(DRQN, self).__init__()
        # Embedding for tile types 0-4
        self.embedding = nn.Embedding(5, embedding_dim)
        
        # Process the 3x3 window (9 tiles)
        self.feature_layer = nn.Sequential(
            nn.Linear(9 * embedding_dim, hidden_dim),
            nn.ReLU()
        )
        
        # LSTM for temporal dependencies (POMDP memory)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Q-value output
        self.fc_q = nn.Linear(hidden_dim, n_actions)

    def forward(self, x, hidden_state=None):
        # x shape: (batch, seq, 3, 3)
        b, s, h, w = x.size()
        
        # Convert to long for embedding, then flatten spatial dimensions
        x = self.embedding(x.long())
        x = x.view(b, s, -1) # (batch, seq, 9 * embedding_dim)
        
        x = self.feature_layer(x)
        lstm_out, new_hidden = self.lstm(x, hidden_state)
        q_values = self.fc_q(lstm_out)
        
        return q_values, new_hidden

    def init_hidden(self, batch_size=1, device="cpu"):
        return (torch.zeros(1, batch_size, 128).to(device),
                torch.zeros(1, batch_size, 128).to(device))

# --- 2. Episode-based Replay Buffer ---
class EpisodeReplayBuffer:
    def __init__(self, capacity=2000, sequence_length=12):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.memory = deque(maxlen=capacity)

    def push(self, episode):
        self.memory.append(episode)

    def sample(self, batch_size):
        sampled_episodes = random.sample(self.memory, batch_size)
        b_obs, b_act, b_rew, b_next_obs, b_done = [], [], [], [], []

        for ep in sampled_episodes:
            # Pick a random window within the episode
            start = random.randint(0, max(0, len(ep) - self.sequence_length))
            seq = ep[start : start + self.sequence_length]
            
            # Extract transitions
            o, a, r, no, d = zip(*seq)
            
            # Pad sequences shorter than sequence_length
            if len(o) < self.sequence_length:
                pad = self.sequence_length - len(o)
                o = list(o) + [np.zeros((3,3))] * pad
                a = list(a) + [0] * pad
                r = list(r) + [0.0] * pad
                no = list(no) + [np.zeros((3,3))] * pad
                d = list(d) + [True] * pad
            
            b_obs.append(o)
            b_act.append(a)
            b_rew.append(r)
            b_next_obs.append(no)
            b_done.append(d)

        return (torch.FloatTensor(np.array(b_obs)), 
                torch.LongTensor(np.array(b_act)),
                torch.FloatTensor(np.array(b_rew)), 
                torch.FloatTensor(np.array(b_next_obs)),
                torch.BoolTensor(np.array(b_done)))

# --- 3. Training Loop ---
def train():
    env = EnhancedFrozenLake()
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Hyperparameters
    BATCH_SIZE = 32
    GAMMA = 0.99
    LR = 1e-4
    EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 0.9995 # Option C: Faster decay
    TARGET_UPDATE = 10 
    MAX_EPISODES = 20000
    MAX_STEPS = 200 # Option B: More time to explore

    policy_net = DRQN().to(device)
    target_net = DRQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    buffer = EpisodeReplayBuffer(capacity=2000, sequence_length=20) # Option B: Longer memory

    
    epsilon = EPS_START

    for ep in range(1, MAX_EPISODES + 1):
        obs, info = env.reset()
        # Initial hidden state for the episode
        hidden = policy_net.init_hidden(1, device=device)
        episode_data = []
        total_reward = 0
        
        for t in range(MAX_STEPS):
            # Select Action
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values, next_hidden = policy_net(obs_tensor, hidden)
            
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                # Use the Q-value of the last step in the sequence (which is our current step)
                action = q_values[0, -1].argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            episode_data.append((obs, action, reward, next_obs, done))
            
            obs = next_obs
            hidden = next_hidden # Carry hidden state forward through episode
            total_reward += reward
            
            if done:
                break
        
        buffer.push(episode_data)
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        # Optimization Step
        if len(buffer.memory) > BATCH_SIZE:
            b_obs, b_act, b_rew, b_next_obs, b_done = buffer.sample(BATCH_SIZE)
            b_obs = b_obs.to(device)
            b_act = b_act.to(device)
            b_rew = b_rew.to(device)
            b_next_obs = b_next_obs.to(device)
            b_done = b_done.to(device)

            # Get Q-values for the entire sampled sequence
            h0 = policy_net.init_hidden(BATCH_SIZE, device=device)
            q_out, _ = policy_net(b_obs, h0)
            
            # Gather Q-values for the actions taken
            q_values = q_out.gather(2, b_act.unsqueeze(2)).squeeze(2)

            with torch.no_grad():
                h_target = target_net.init_hidden(BATCH_SIZE, device=device)
                next_q_out, _ = target_net(b_next_obs, h_target)
                max_next_q = next_q_out.max(dim=2)[0]
                targets = b_rew + (GAMMA * max_next_q * (~b_done))

            loss = F.smooth_l1_loss(q_values, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
            optimizer.step()

        # Update Target Network
        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Logging
        if ep % 100 == 0:
            print(f"Ep: {ep} | Avg Reward (this ep): {total_reward:.2f} | Epsilon: {epsilon:.3f} | Wind: {info['wind']}")
    
    # Save the final model
    torch.save(policy_net.state_dict(), "drqn_frozenlake.pth")
    print("Model saved to drqn_frozenlake.pth")

def test_and_render(model_path="drqn_frozenlake.pth"):
    env = EnhancedFrozenLake(render_mode="ansi")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DRQN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs, info = env.reset()
    hidden = model.init_hidden(1, device=device)
    done = False
    total_reward = 0

    print("\n--- Visualizing Trained Agent ---")
    while not done:
        print(env.render())
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values, hidden = model(obs_tensor, hidden)
        
        action = q_values[0, -1].argmax().item()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    
    print(env.render())
    print(f"Final Reward: {total_reward:.2f}")

if __name__ == "__main__":
    # To train, keep this:
    train()
    # To watch the agent after training, uncomment this:
    # test_and_render()
