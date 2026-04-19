import torch
import numpy as np
import glob
import os
from enhanced_frozen_lake import EnhancedFrozenLake
from drqn_agent import DRQN
from qr_drqn_agent import QR_DRQN
from q_learning_agent import QLearningAgent

def get_optimal_bfs_path_length(env):
    start = (0, 0)
    goal = (env.size - 1, env.size - 1)
    queue = [(start, 0)]
    visited = {start}
    
    while queue:
        (r, c), steps = queue.pop(0)
        
        if (r, c) == goal:
            return steps
            
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < env.size and 0 <= nc < env.size:
                if env.grid[nr, nc] != env.TILE_TYPES['H'] and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), steps + 1))
    return 0

def evaluate_models(num_episodes=100):
    all_files = glob.glob("map8_v5_*.pth") + glob.glob("map8_v5_*.pkl")
    if not all_files:
        print("No models found in map8_v5 directory.")
        return
        
    # Group models by method and find the latest for each
    methods = {}
    for f in all_files:
        fname = os.path.basename(f)
        if "QR-DRQN" in fname:
            method = "QR-DRQN"
        elif "RND-DRQN" in fname:
            method = "RND-DRQN"
        elif "Q-Learning" in fname:
            method = "Tabular Q-Learning"
        elif "DRQN" in fname:
            method = "Standard DRQN"
        else:
            method = "Legacy (Ignore)"
            
        if method not in methods:
            methods[method] = f
        else:
            if os.path.getctime(f) > os.path.getctime(methods[method]):
                methods[method] = f
                
    files = [f for m, f in methods.items() if m != "Legacy (Ignore)"]
    
    env = EnhancedFrozenLake()
    
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")

    results = []
    print(f"Evaluating {len(files)} models over {num_episodes} episodes each...\n")

    # ----- Oracle (BFS No-Wind) Baseline -----
    print("Calculating Oracle (BFS) Baseline...")
    oracle_rewards = 0
    oracle_steps = []
    for ep in range(num_episodes):
        env.reset(seed=ep+2000)
        steps = get_optimal_bfs_path_length(env)
        # Starting dist is 14 (7+7). 14 * 0.5 (progress bonus) = 7.0 
        # Actually in step(), if dist_after < min_dist_this_ep, reward += 0.1
        # The max dist is 14, min dist is 0. So it gets +0.1 exactly 14 times. = +1.4
        reward = 30.0 + 1.4 - (steps * 0.05)
        oracle_rewards += reward
        oracle_steps.append(steps)
    
    results.append({
        'name': "Oracle (BFS - Perfect Path)",
        'sr': 100.0,
        'avg_rew': oracle_rewards / num_episodes,
        'avg_steps': np.mean(oracle_steps)
    })

    for model_path in files:
        filename = os.path.basename(model_path)
        is_qr = "QR-DRQN" in filename
        is_rnd = "RND-DRQN" in filename
        is_qlearn = "Q-Learning" in filename
        
        if is_qlearn:
            model = QLearningAgent()
            model.load(model_path)
        elif is_qr:
            model = QR_DRQN().to(device)
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
            except RuntimeError as e:
                print(f"Skipping {filename}: Incompatible model architecture.")
                continue
            model.eval()
        else:
            # Both standard DRQN and RND-DRQN use the same underlying policy architecture
            model = DRQN().to(device)
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
            except RuntimeError as e:
                print(f"Skipping {filename}: Incompatible model architecture.")
                continue
            model.eval()
        
        successes = 0
        total_rewards = 0
        steps_list = []
        
        for ep in range(num_episodes):
            obs, info = env.reset(seed=ep+2000) # Fixed seeds for fairness
            wind = info['wind']
            if not is_qlearn:
                hidden = model.init_hidden(1, device=device)
            done = False
            step_count = 0
            ep_reward = 0
            
            while not done:
                if is_qlearn:
                    action = model.get_action(obs, epsilon=0.0)
                else:
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