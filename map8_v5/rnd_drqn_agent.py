import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from datetime import datetime
from enhanced_frozen_lake import EnhancedFrozenLake
from drqn_agent import DRQN, EpisodeReplayBuffer

class RNDNetwork(nn.Module):
    def __init__(self, embedding_dim=16, hidden_dim=64, output_dim=32):
        super(RNDNetwork, self).__init__()
        self.vision_embedding = nn.Embedding(5, embedding_dim)
        self.wind_embedding = nn.Embedding(4, embedding_dim)
        self.net = nn.Sequential(
            nn.Linear((25 + 1) * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, vision, wind):
        b, s, h, w = vision.size()
        v_feat = self.vision_embedding(vision.long()).view(b, s, -1)
        w_feat = self.wind_embedding(wind.long())
        combined = torch.cat([v_feat, w_feat], dim=2)
        return self.net(combined)

def evaluate(model, device, num_episodes=20):
    env = EnhancedFrozenLake()
    success_count = 0
    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep+1000)
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
            if terminated and reward > 10: success_count += 1
            done = terminated or truncated
    return (success_count / num_episodes) * 100

def train():
    env = EnhancedFrozenLake()
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    
    print(f"Training on device: {device} | map8_v5 RND-DRQN Multi-Input")
    
    BATCH_SIZE = 32
    GAMMA = 0.99
    LR = 1e-4
    EPS_DECAY = 0.99995
    MAX_EPISODES = 150000 
    INT_REWARD_WEIGHT = 0.5 # Curiosity weight
    
    policy_net = DRQN().to(device)
    target_net = DRQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    rnd_target = RNDNetwork().to(device)
    rnd_predictor = RNDNetwork().to(device)
    # Freeze RND target
    for param in rnd_target.parameters(): 
        param.requires_grad = False
    
    optimizer = optim.Adam(list(policy_net.parameters()) + list(rnd_predictor.parameters()), lr=LR)
    buffer = EpisodeReplayBuffer(capacity=5000, sequence_length=20)
    epsilon = 1.0
    
    reward_history = deque(maxlen=100)

    for ep in range(1, MAX_EPISODES + 1):
        obs, info = env.reset()
        wind = info['wind']
        hidden = policy_net.init_hidden(1, device=device)
        episode_data = []
        total_ext_reward = 0
        
        for t in range(200):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(device)
            wind_t = torch.LongTensor([wind]).unsqueeze(0).to(device)
            
            with torch.no_grad():
                q_values, next_hidden = policy_net(obs_t, wind_t, hidden)
                # Compute intrinsic reward
                pred_feat = rnd_predictor(obs_t, wind_t)
                targ_feat = rnd_target(obs_t, wind_t)
                int_reward = F.mse_loss(pred_feat, targ_feat).item()
                
            action = env.action_space.sample() if random.random() < epsilon else q_values[0, -1].argmax().item()
            next_obs, ext_reward, terminated, truncated, info = env.step(action)
            next_wind = info.get('wind', wind)
            
            # Combine reward
            combined_reward = ext_reward + INT_REWARD_WEIGHT * int_reward
            
            done = terminated or truncated
            episode_data.append((obs, wind, action, combined_reward, next_obs, next_wind, done))
            obs, wind, hidden = next_obs, next_wind, next_hidden
            total_ext_reward += ext_reward
            if done: break
        
        buffer.push(episode_data)
        reward_history.append(total_ext_reward) # Track EXTRINSIC reward for human readability
        epsilon = max(0.01, epsilon * EPS_DECAY)

        if len(buffer.memory) > BATCH_SIZE:
            b_obs, b_wind, b_act, b_rew, b_nobs, b_nwind, b_done = [x.to(device) for x in buffer.sample(BATCH_SIZE)]
            
            # RND Loss
            pred_feat = rnd_predictor(b_obs, b_wind)
            with torch.no_grad(): targ_feat = rnd_target(b_obs, b_wind)
            rnd_loss = F.mse_loss(pred_feat, targ_feat)
            
            # DRQN Loss
            q_out, _ = policy_net(b_obs, b_wind, policy_net.init_hidden(BATCH_SIZE, device))
            q_values = q_out.gather(2, b_act.unsqueeze(2)).squeeze(2)
            with torch.no_grad():
                next_q_out, _ = target_net(b_nobs, b_nwind, target_net.init_hidden(BATCH_SIZE, device))
                targets = b_rew + (GAMMA * next_q_out.max(dim=2)[0] * (~b_done))
            drqn_loss = F.smooth_l1_loss(q_values, targets)
            
            loss = drqn_loss + rnd_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(rnd_predictor.parameters(), 1.0)
            optimizer.step()

        if ep % 20 == 0: target_net.load_state_dict(policy_net.state_dict())
        
        if ep % 500 == 0:
            eval_sr = evaluate(policy_net, device)
            print(f"Ep: {ep} | Train Ext Avg Rew: {np.mean(reward_history):.2f} | Eval SR: {eval_sr:.1f}% | Eps: {epsilon:.3f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"map8_v5_RND-DRQN_{ts}.pth"
    torch.save(policy_net.state_dict(), filename) # We only need to save policy_net to play
    print(f"Model saved: {filename}")

if __name__ == "__main__":
    train()
