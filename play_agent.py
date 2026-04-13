import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time
from enhanced_frozen_lake import EnhancedFrozenLake
from drqn_agent import DRQN

def play_with_visualization():
    # 1. Setup Environment and Device
    env = EnhancedFrozenLake()
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    
    # 2. Load the Trained Model
    model_path = "drqn_frozenlake.pth"
    model = DRQN().to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Please train first.")
        return

    model.eval()

    # 3. Graphical Setup
    plt.ion()
    fig, (ax_global, ax_local) = plt.subplots(1, 2, figsize=(14, 7))
    
    # 0:F (Blue), 1:S (Green), 2:G (Gold), 3:H (Red), 4:W (Gray)
    cmap = colors.ListedColormap(['#a0d8f1', '#77dd77', '#fdfd96', '#ff6961', '#333333'])
    norm = colors.BoundaryNorm([0, 1, 2, 3, 4, 5], cmap.N)

    # Action Mapping for arrows: 0:L, 1:D, 2:R, 3:U
    action_arrows = {0: (-0.4, 0), 1: (0, 0.4), 2: (0.4, 0), 3: (0, -0.4)}
    action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
    wind_arrows = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

    for ep in range(5):
        obs, info = env.reset()
        hidden = model.init_hidden(1, device=device)
        done = False
        total_reward = 0
        step_count = 0
        current_wind = info['wind']
        last_action = None
        last_reward = 0

        while not done:
            # --- Rendering Logic ---
            ax_global.clear()
            ax_global.imshow(env.grid, cmap=cmap, norm=norm)
            ax_global.plot(env.agent_pos[1], env.agent_pos[0], 'go', markersize=18, markeredgecolor='white', zorder=5)
            
            # Show Wind Direction as a big arrow in the corner
            wx, wy = wind_arrows[current_wind]
            ax_global.arrow(0.5, 0.5, wx*0.5, wy*0.5, head_width=0.2, color='white', alpha=0.6, transform=ax_global.transAxes)
            ax_global.text(0.5, 0.3, "WIND", color='white', ha='center', transform=ax_global.transAxes, fontweight='bold')
            
            ax_global.set_title(f"EPISODE {ep+1} | STEP {step_count}\nTotal Reward: {total_reward:.2f}")

            ax_local.clear()
            ax_local.imshow(obs, cmap=cmap, norm=norm)
            ax_local.plot(1, 1, 'go', markersize=25, markeredgecolor='white')
            
            # Show Decision Arrow on the Local View
            if last_action is not None:
                dx, dy = action_arrows[last_action]
                ax_local.arrow(1, 1, dx, dy, head_width=0.15, fc='yellow', ec='black', lw=2, zorder=10)
                ax_local.set_xlabel(f"DECISION: {action_names[last_action]}", fontsize=12, fontweight='bold', color='darkorange')

            ax_local.set_title(f"AGENT'S 3x3 VISION\nStep Reward: {last_reward:+.2f}")

            # Styling
            for ax in [ax_global, ax_local]:
                size = 8 if ax == ax_global else 3
                ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
                ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
                ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

            plt.draw()
            plt.pause(0.15) # Speed of the "movie"

            # --- Agent Logic ---
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values, hidden = model(obs_tensor, hidden)
            
            action = q_values[0, -1].argmax().item()
            last_action = action
            
            obs, reward, terminated, truncated, _ = env.step(action)
            last_reward = reward
            total_reward += reward
            step_count += 1
            done = terminated or truncated

        # End of Episode pause
        msg = "SUCCESS" if total_reward > 4.0 else "HOLE!"
        color = 'green' if total_reward > 4.0 else 'red'
        ax_global.text(0.5, 0.5, msg, transform=ax_global.transAxes, fontsize=50, 
                       color=color, alpha=0.8, ha='center', va='center', fontweight='bold')
        plt.draw()
        plt.pause(1.5)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    play_with_visualization()
