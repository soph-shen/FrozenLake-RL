import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time
from enhanced_frozen_lake import EnhancedFrozenLake
from drqn_agent import DRQN

def visualize():
    # 1. Setup
    env = EnhancedFrozenLake()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DRQN().to(device)
    try:
        model.load_state_dict(torch.load("drqn_frozenlake.pth", map_location=device))
    except:
        print("Model not found. Train it first!")
        return
    model.eval()

    # 2. Color Map Setup
    # 0:F, 1:S, 2:G, 3:H, 4:W
    cmap = colors.ListedColormap(['#a0d8f1', '#77dd77', '#fdfd96', '#ff6961', '#333333'])
    bounds = [0, 1, 2, 3, 4, 5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # 3. Initialize Plot
    plt.ion() # Interactive mode
    fig, (ax_global, ax_local) = plt.subplots(1, 2, figsize=(12, 6))
    
    obs, info = env.reset()
    hidden = model.init_hidden(1, device=device)
    done = False
    total_reward = 0

    while not done:
        # --- A. Render Global Map ---
        ax_global.clear()
        # Create a display grid where the agent is marked
        display_grid = env.grid.copy()
        ax_global.imshow(display_grid, cmap=cmap, norm=norm)
        
        # Draw Agent Position
        r, c = env.agent_pos
        ax_global.plot(c, r, 'go', markersize=15, markeredgecolor='white', label='Agent')
        ax_global.set_title(f"Global Map (Wind: {info['wind']})")
        
        # --- B. Render Local 3x3 Observation ---
        ax_local.clear()
        ax_local.imshow(obs, cmap=cmap, norm=norm)
        # Mark center of 3x3 (the agent)
        ax_local.plot(1, 1, 'go', markersize=20, markeredgecolor='white')
        ax_local.set_title("Agent's 3x3 View (POMDP)")

        # Cleanup axes
        for ax in [ax_global, ax_local]:
            ax.set_xticks(np.arange(-0.5, 8.5 if ax == ax_global else 3.5, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 8.5 if ax == ax_global else 3.5, 1), minor=True)
            ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
            ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        plt.draw()
        plt.pause(0.3)

        # --- C. Agent Decision ---
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values, hidden = model(obs_tensor, hidden)
        
        action = q_values[0, -1].argmax().item()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    # Final result message
    plt.ioff()
    result_text = "SUCCESS!" if total_reward > 4.0 else "FAILURE!"
    ax_global.set_title(f"Result: {result_text} (Reward: {total_reward:.2f})", color='green' if total_reward > 4.0 else 'red', fontsize=14)
    plt.show()

if __name__ == "__main__":
    visualize()
