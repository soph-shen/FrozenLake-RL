import torch
import numpy as np
import glob
import os
from enhanced_frozen_lake import EnhancedFrozenLake
from drqn_agent import DRQN
from qr_drqn_agent import QR_DRQN

def evaluate_models(num_episodes=100):
    files = glob.glob("map8_v5_*.pth")
    if not files:
        print("No models found in map8_v5 directory.")
        return
        
    env = EnhancedFrozenLake()
    
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")

    results = []
    print(f"Evaluating {len(files)} models over {num_episodes} episodes each...\n")

    for model_path in files:
        filename = os.path.basename(model_path)
        is_qr = "QR-DRQN" in filename
        is_rnd = "RND-DRQN" in filename
        
        if is_qr:
            model = QR_DRQN().to(device)
        else:
            # Both standard DRQN and RND-DRQN use the same underlying policy architecture
            model = DRQN().to(device)
            
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except RuntimeError as e:
            print(f"Skipping {filename}: Incompatible model architecture (Likely an older version).")
            continue
            
        model.eval()
        
        successes = 0
        total_rewards = 0
        steps_list = []
        
        for ep in range(num_episodes):
            obs, info = env.reset(seed=ep+2000) # Fixed seeds for fairness
            wind = info['wind']
            hidden = model.init_hidden(1, device=device)
            done = False
            step_count = 0
            ep_reward = 0
            
            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(device)
                wind_t = torch.LongTensor([wind]).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    if is_qr:
                        out, hidden = model(obs_t, wind_t, hidden)
                        q_values = out.mean(dim=3)
                    else:
                        q_values, hidden = model(obs_t, wind_t, hidden)
                        
                action = q_values[0, -1].argmax().item()
                obs, reward, terminated, truncated, info = env.step(action)
                wind = info.get('wind', wind)
                ep_reward += reward
                step_count += 1
                
                if terminated and reward > 10:
                    successes += 1
                    steps_list.append(step_count)
                    
                done = terminated or truncated
                
            total_rewards += ep_reward
            
        sr = (successes / num_episodes) * 100
        avg_rew = total_rewards / num_episodes
        avg_steps = np.mean(steps_list) if steps_list else 0
        
        results.append({
            'name': filename,
            'sr': sr,
            'avg_rew': avg_rew,
            'avg_steps': avg_steps
        })

    # Print Results Table
    print("-" * 85)
    print(f"{'Model Name':<35} | {'Success Rate':<12} | {'Avg Reward':<12} | {'Avg Steps (Success)'}")
    print("-" * 85)
    for r in sorted(results, key=lambda x: x['sr'], reverse=True):
        print(f"{r['name']:<35} | {r['sr']:>11.1f}% | {r['avg_rew']:>12.2f} | {r['avg_steps']:>10.1f}")
    print("-" * 85)

if __name__ == "__main__":
    evaluate_models()
