import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time
import os
import glob
from enhanced_frozen_lake import EnhancedFrozenLake
from drqn_agent import DRQN

def get_latest_model():
    files = glob.glob("map16_v7_*.pth")
    if not files: return None
    return max(files, key=os.path.getctime)

def play():
    # 1. Setup Environment
    env = EnhancedFrozenLake()
    # Set to TARGET difficulty for final validation
    env.num_holes = 25
    env.wind_probability = 0.1
    
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    
    latest_model = get_latest_model()
    if not latest_model:
        print("No model found. Please train first.")
        return
    
    print(f"Loading latest model: {latest_model} on {device}")
    # Using the 256-unit "Power-Up" brain
    model = DRQN(hidden_dim=256).to(device)
    model.load_state_dict(torch.load(latest_model, map_location=device))
    model.eval()

    plt.ion()
    fig, (ax_global, ax_local) = plt.subplots(1, 2, figsize=(14, 7))
    cmap = colors.ListedColormap(['#a0d8f1', '#77dd77', '#fdfd96', '#ff6961', '#333333'])
    norm = colors.BoundaryNorm([0, 1, 2, 3, 4, 5], cmap.N)

    action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
    wind_arrows = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

    for ep in range(5):
        obs, info = env.reset()
        wind = info['wind']
        hidden = model.init_hidden(1, device=device)
        done, total_reward, step_count = False, 0, 0
        last_action = None

        while not done:
            ax_global.clear()
            ax_global.imshow(env.grid, cmap=cmap, norm=norm)
            ax_global.plot(env.agent_pos[1], env.agent_pos[0], 'go', markersize=12, markeredgecolor='white')
            
            # Wind indicator
            wx, wy = wind_arrows[wind]
            ax_global.arrow(0.85, 0.85, wx*0.1, -wy*0.1, head_width=0.05, color='white', 
                            linewidth=3, transform=ax_global.transAxes, zorder=10)
            ax_global.text(0.85, 0.75, f"WIND: {action_names[wind]}", color='white', 
                           fontsize=10, fontweight='bold', ha='center', transform=ax_global.transAxes,
                           bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
            
            ax_global.set_title(f"EP {ep+1} | GLOBAL (16x16)\nTotal Reward: {total_reward:.2f}")

            ax_local.clear()
            ax_local.imshow(obs, cmap=cmap, norm=norm)
            ax_local.plot(3, 3, 'go', markersize=20, markeredgecolor='white') 
            ax_local.set_title(f"AGENT VISION (7x7)\nLast Action: {action_names.get(last_action, 'None')}")

            for ax in [ax_global, ax_local]:
                size = 16 if ax == ax_global else 7
                ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
                ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
                ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

            plt.draw(); plt.pause(0.1)

            obs_t = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(device)
            wind_t = torch.LongTensor([wind]).unsqueeze(0).to(device)
            
            with torch.no_grad():
                q_values, hidden = model(obs_t, wind_t, hidden)
            
            action = q_values[0, -1].argmax().item()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            last_action = action
            total_reward, step_count, obs = total_reward + reward, step_count + 1, next_obs
            done = terminated or truncated

        msg = "SUCCESS" if total_reward > 10.0 else "FAIL"
        color = 'green' if total_reward > 10.0 else 'red'
        ax_global.text(0.5, 0.5, msg, transform=ax_global.transAxes, fontsize=40, color=color, ha='center', fontweight='bold')
        plt.draw(); plt.pause(1.5)

if __name__ == "__main__":
    play()
