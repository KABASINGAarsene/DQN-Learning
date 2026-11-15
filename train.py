"""
train.py - Deep Q-Network (DQN) Agent for Atari Pong
=====================================================
Assignment: Formative 2 — Implementing a DQN agent from scratch.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import ale_py                   
from collections import deque
import matplotlib.pyplot as plt 

gym.register_envs(ale_py)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
ENV_NAME = "ALE/Pong-v5"       
FRAME_STACK = 4                 
BUFFER_CAPACITY = 10_000        
BATCH_SIZE = 32                 
LEARNING_RATE = 0.001           
GAMMA = 0.99                    
EPSILON_START = 1.0             
EPSILON_END = 0.05              
EPSILON_DECAY = 0.995           
NUM_EPISODES = 500              
TARGET_UPDATE_FREQ = 10         
SAVE_PATH = "dqn_model_exp1.pth"    

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — ENVIRONMENT INITIALIZATION
# Prepares the Gymnasium Atari environment.
# Resizes frames to 84x84, converts to greyscale, and stacks the last 4 frames.
# ─────────────────────────────────────────────────────────────────────────────
def make_env():
    env = gym.make(ENV_NAME, render_mode=None)
    env = gym.wrappers.ResizeObservation(env, shape=(84, 84))
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
    env = gym.wrappers.FrameStackObservation(env, stack_size=FRAME_STACK)
    return env

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Q-NETWORK (CNN)
# CNN architecture mapping stacked pixel observations to Q-values for each action.
# Uses 3 Conv2d layers followed by 2 Linear layers.
# ─────────────────────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, num_actions):
        super(QNetwork, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=FRAME_STACK, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=64 * 7 * 7, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_actions),
        )

    def forward(self, pixel_frames):
        conv_output = self.conv_layers(pixel_frames)
        flat = conv_output.view(conv_output.size(0), -1)
        q_values = self.fc_layers(flat)
        return q_values

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — EXPERIENCE REPLAY BUFFER
# Circular buffer storing past transitions (state, action, reward, next_state, done).
# Used to sample random mini-batches for training to break data correlation.
# ─────────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(DEVICE) / 255.0
        actions_tensor = torch.tensor(np.array(actions), dtype=torch.int64).to(DEVICE)
        rewards_tensor = torch.tensor(np.array(rewards), dtype=torch.float32).to(DEVICE)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32).to(DEVICE) / 255.0
        dones_tensor = torch.tensor(np.array(dones, dtype=np.float32), dtype=torch.float32).to(DEVICE)

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor

    def __len__(self):
        return len(self.buffer)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — EPSILON-GREEDY ACTION SELECTION
# Balances exploration (random actions) and exploitation (highest Q-value).
# ─────────────────────────────────────────────────────────────────────────────
def select_action(state, online_network, epsilon, num_actions):
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)
    
    with torch.no_grad():
        state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(DEVICE) / 255.0          
        q_values = online_network(state_tensor)     
        
    return q_values.argmax(dim=1).item()            

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — OPTIMIZE MODEL (BELLMAN EQUATION + LOSS + BACKPROP)
# Computes Huber Loss between online Q-values and target Q-values.
# Updates the online network weights via backpropagation.
# ─────────────────────────────────────────────────────────────────────────────
def optimize_model(online_network, target_network, replay_buffer, optimizer):
    if len(replay_buffer) < BATCH_SIZE:
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    predicted_q_values = online_network(states).gather(
        dim=1, index=actions.unsqueeze(1)
    ).squeeze(1)                     

    with torch.no_grad():
        best_next_q_values = target_network(next_states).max(dim=1).values
        target_q_values = rewards + GAMMA * best_next_q_values * (1.0 - dones)

    loss_fn = nn.SmoothL1Loss()
    loss = loss_fn(predicted_q_values, target_q_values)

    optimizer.zero_grad()   
    loss.backward()         
    optimizer.step()        

    return loss.item()      

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — MAIN TRAINING LOOP
# Orchestrates environment resets, epsilon decay, target network updates, 
# and logging over NUM_EPISODES.
# ─────────────────────────────────────────────────────────────────────────────
def main():
    env = make_env()
    num_actions = env.action_space.n    

    online_network = QNetwork(num_actions=num_actions).to(DEVICE)
    target_network = QNetwork(num_actions=num_actions).to(DEVICE)
    target_network.load_state_dict(online_network.state_dict())
    target_network.eval()   

    optimizer = torch.optim.Adam(online_network.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)
    epsilon = EPSILON_START

    episode_rewards = []    
    episode_lengths = []    

    print(f"Starting training for {NUM_EPISODES} episodes on {DEVICE}...")
    print(f"Actions available: {num_actions}")
    print("-" * 50)

    for episode in range(1, NUM_EPISODES + 1):
        obs, _ = env.reset()
        state = np.array(obs)       

        episode_reward = 0.0        
        episode_loss   = 0.0        
        step_count     = 0          

        done = False
        while not done:
            action = select_action(state, online_network, epsilon, num_actions)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_obs)
            done = terminated or truncated

            replay_buffer.store(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            step_count += 1

            loss = optimize_model(online_network, target_network, replay_buffer, optimizer)
            if loss is not None:
                episode_loss += loss

        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if episode % TARGET_UPDATE_FREQ == 0:
            target_network.load_state_dict(online_network.state_dict())

        if episode % 1 == 0:
            avg_reward = np.mean(episode_rewards[-10:])  
            avg_loss   = episode_loss / max(step_count, 1)
            print(f"Episode {episode:4d}/{NUM_EPISODES} | "
                  f"Avg Reward (last 10): {avg_reward:6.1f} | "
                  f"Epsilon: {epsilon:.3f} | "
                  f"Avg Loss: {avg_loss:.4f} | "
                  f"Buffer: {len(replay_buffer)}")

    torch.save(online_network.state_dict(), SAVE_PATH)
    print("-" * 50)
    print(f"Training complete! Model saved to '{SAVE_PATH}'")

    print("Generating training plots...")
    plot_path = SAVE_PATH.replace('.pth', '.png')  
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, color="blue", alpha=0.7)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(episode_lengths, color="orange", alpha=0.7)
    plt.title("Episode Lengths")
    plt.xlabel("Episode")
    plt.ylabel("Number of Steps")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Plots saved to '{plot_path}'")
    env.close()

if __name__ == "__main__":
    main()