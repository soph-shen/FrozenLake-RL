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
    files = glob.glob("map8_v5_*.pth")
    if not files: return None
    return max(files, key=os.path.getctime)

def play():
    # 1. Setup Environment and Device
    env = EnhancedFrozenLake()
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # 2. Dynamic Model Loading
    latest_model = get_latest_model()
    if not latest_model:
        print("No model found in map8_v5 folder. Please train first.")
        return
    
    print(f"Loading latest model: {latest_model} on {device}")
    model = DRQN().to(device)
    model.load_state_dict(torch.load(latest_model, map_location=device))
    model.eval()

    # 3. Graphical Setup
    plt.ion()
    fig, (ax_global, ax_local) = plt.subplots(1, 2, figsize=(14, 7))
    
    # 0:F (Blue), 1:S (Green), 2:G (Gold), 3:H (Red), 4:W (Gray)
    cmap = colors.ListedColormap(['#a0d8f1', '#77dd77', '#fdfd96', '#ff6961', '#333333'])
    norm = colors.BoundaryNorm([0, 1, 2, 3, 4, 5], cmap.N)

    # UI Mapping
    action_arrows = {0: (-0.4, 0), 1: (0, 0.4), 2: (0.4, 0), 3: (0, -0.4)}
    action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
    wind_arrows = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

    for ep in range(5):
        obs, info = env.reset()
        wind = info['wind']
        hidden = model.init_hidden(1, device=device)
        done, total_reward, step_count = False, 0, 0
        last_action, last_reward = None, 0

        while not done:
            # --- Rendering ---
            ax_global.clear()
            ax_global.imshow(env.grid, cmap=cmap, norm=norm)
            ax_global.plot(env.agent_pos[1], env.agent_pos[0], 'go', markersize=18, markeredgecolor='white', zorder=5)
            
            # CLEAR WIND INDICATOR
            wx, wy = wind_arrows[wind]
            # Place arrow in the top-right corner area
            ax_global.arrow(0.85, 0.85, wx*0.1, -wy*0.1, head_width=0.05, color='white', 
                            linewidth=3, transform=ax_global.transAxes, zorder=10)
            ax_global.text(0.85, 0.75, f"WIND: {action_names[wind]}", color='white', 
                           fontsize=10, fontweight='bold', ha='center', transform=ax_global.transAxes,
                           bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
            
            ax_global.set_title(f"EPISODE {ep+1} | STEP {step_count}\nTotal Reward: {total_reward:.2f}")

            ax_local.clear()
            ax_local.imshow(obs, cmap=cmap, norm=norm)
            ax_local.plot(2, 2, 'go', markersize=25, markeredgecolor='white') 
            
            if last_action is not None:
                dx, dy = action_arrows[last_action]
                ax_local.arrow(2, 2, dx, dy, head_width=0.15, fc='yellow', ec='black', lw=2, zorder=10)
                ax_local.set_xlabel(f"DECISION: {action_names[last_action]}", fontsize=12, fontweight='bold', color='darkorange')

            ax_local.set_title(f"AGENT'S 5x5 VISION\nStep Reward: {last_reward:+.2f}")

            for ax in [ax_global, ax_local]:
                size = 8 if ax == ax_global else 5
                ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
                ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
                ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

            plt.draw(); plt.pause(0.15)

            # --- Agent Logic ---
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(device)
            wind_tensor = torch.LongTensor([wind]).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # Correct call with 3 arguments: vision, wind, hidden
                q_values, hidden = model(obs_tensor, wind_tensor, hidden)
            
            action = q_values[0, -1].argmax().item()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            last_action, last_reward = action, reward
            obs, total_reward, step_count = next_obs, total_reward + reward, step_count + 1
            done = terminated or truncated

        msg = "SUCCESS" if total_reward > 10.0 else "FAIL"
        color = 'green' if total_reward > 10.0 else 'red'
        ax_global.text(0.5, 0.5, msg, transform=ax_global.transAxes, fontsize=40, color=color, alpha=0.8, ha='center', va='center', fontweight='bold')
        plt.draw(); plt.pause(1.5)

    plt.ioff(); plt.show()

if __name__ == "__main__":
    play()
