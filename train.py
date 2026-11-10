"""
train.py - Deep Q-Network (DQN) Agent for Atari Pong
=====================================================
Assignment: Formative 2 — Implementing a DQN agent from scratch.

This file is intentionally written to be easy to read and explain.
Every major section has a plain-English comment block above it.
We follow the original DQN paper (Mnih et al., 2015) closely.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import ale_py                   # Registers ALE/Atari environments with Gymnasium
from collections import deque
import matplotlib.pyplot as plt # Added for rubric-required plots

# Tell Gymnasium about all the ALE (Atari) environments, including Pong.
# Without this line, gym.make('ALE/Pong-v5') raises a NamespaceNotFound error.
gym.register_envs(ale_py)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — HYPERPARAMETERS
# These are the knobs your teammates will tweak during their experiments.
# They are all gathered here in one place so nothing is buried in the code.
# ─────────────────────────────────────────────────────────────────────────────

# --- Environment ---
ENV_NAME = "ALE/Pong-v5"       # The Atari Pong game we are training on

# --- Frame stacking ---
# Instead of feeding the agent a single frame, we stack the last 4 frames
# together. This gives the network a sense of motion (e.g., which way the
# ball is moving). Without stacking, the agent would be "blind" to velocity.
FRAME_STACK = 4                 # Number of consecutive frames stacked together

# --- Replay Buffer ---
BUFFER_CAPACITY = 10_000        # Max number of past experiences we keep
                                # (kept small so it fits in RAM easily)
BATCH_SIZE = 32                 # How many experiences we sample per training step

# --- Learning ---
LEARNING_RATE = 0.001           # Step size for the Adam optimizer (teammate can tune)
GAMMA = 0.99                    # Discount factor: how much we value future rewards
                                # 0 = only care about immediate reward
                                # 1 = care equally about all future rewards

# --- Epsilon-greedy exploration ---
EPSILON_START = 1.0             # Start fully random (100 % exploration)
EPSILON_END = 0.05              # Never go below 5 % random (always explore a little)
EPSILON_DECAY = 0.995           # Multiply epsilon by this after every episode

# --- Training loop ---
NUM_EPISODES = 500              # Total number of Pong games to play during training
TARGET_UPDATE_FREQ = 10         # Copy online weights → target network every N episodes

# --- Output ---
SAVE_PATH = "dqn_model_exp1.pth"    # File name for the saved model weights

# Automatically use GPU if available, otherwise fall back to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — ENVIRONMENT INITIALIZATION
#
# We use Gymnasium to create the Pong environment.
# Gymnasium provides a standard interface:
#   env.reset()         → returns the starting frame(s)
#   env.step(action)    → returns (next_frame, reward, terminated, truncated, info)
#   env.action_space    → tells us how many actions exist (Pong has 6, but only
#                         3 matter: NOOP, UP, DOWN)
#
# We apply three standard Atari "wrappers" to pre-process the raw game frames:
#   1. ResizeObservation  → shrink the 210×160 colour frame down to 84×84 pixels
#   2. GrayscaleObservation → convert RGB to greyscale (1 colour channel instead of 3)
#   3. FrameStackObservation → stack the last FRAME_STACK frames into one observation
#
# After wrapping, every observation the agent sees has shape:
#   (FRAME_STACK, 84, 84)  →  e.g.  (4, 84, 84)
#   └── 4 stacked greyscale frames, each 84×84 pixels
# ─────────────────────────────────────────────────────────────────────────────

def make_env():
    """
    Creates and returns the pre-processed Pong environment.

    Output shape of each observation: (FRAME_STACK, 84, 84)
    """
    # Create the raw Atari environment
    # render_mode=None means no visual window — faster for training
    env = gym.make(ENV_NAME, render_mode=None)

    # Wrapper 1: Resize each frame from 210×160 → 84×84 pixels.
    # Smaller frames = fewer inputs for the CNN = much faster training.
    env = gym.wrappers.ResizeObservation(env, shape=(84, 84))

    # Wrapper 2: Convert colour (RGB) frames to greyscale.
    # Colour is not needed to understand Pong, so we drop 2 of the 3 channels.
    # This reduces the data per frame by 3×.
    # keep_dim=False → output shape per frame is (84, 84), no extra channel axis.
    # After FrameStackObservation this gives us (4, 84, 84) — exactly what
    # PyTorch's Conv2d expects (channels_first format).
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)

    # Wrapper 3: Stack the last FRAME_STACK frames into a single observation.
    # This gives the agent temporal information (motion of the ball and paddle).
    # Without this, a single frozen frame looks the same whether the ball is
    # moving left or right.
    env = gym.wrappers.FrameStackObservation(env, stack_size=FRAME_STACK)

    return env


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Q-NETWORK (CNN)
#
# This is the "brain" of the DQN agent. It is a Convolutional Neural Network
# (CNN) that takes in the stacked pixel frames and outputs a Q-value for
# every possible action.
#
# What is a Q-value?
#   Q(state, action) = the expected total future reward if we take `action`
#   in `state` and then play optimally from there onwards.
#   The agent always picks the action with the HIGHEST Q-value.
#
# Network INPUT:
#   A batch of stacked frame observations.
#   Shape: (batch_size, FRAME_STACK, 84, 84)
#   e.g.   (32, 4, 84, 84)  — 32 samples, 4 greyscale frames, 84×84 pixels each.
#
# Network OUTPUT:
#   One Q-value per possible action.
#   Shape: (batch_size, num_actions)
#   e.g.   (32, 6)  — 32 samples, 6 possible Pong actions.
#
# Architecture (same as the original DQN paper by Mnih et al., 2015):
#   Conv Layer 1 → Conv Layer 2 → Conv Layer 3 → Flatten → FC Layer → FC Output
#
# The convolutional layers detect spatial patterns in the pixels
# (edges, shapes, objects). The fully connected layers then map those patterns
# to action quality scores (Q-values).
# ─────────────────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    Convolutional Neural Network that maps pixel observations → Q-values.

    Input:  (batch_size, FRAME_STACK, 84, 84)  — stacked greyscale frames
    Output: (batch_size, num_actions)           — one Q-value per action
    """

    def __init__(self, num_actions):
        """
        num_actions: the number of discrete actions available in the environment.
                     For Pong this is 6 (but only 3 are meaningfully different).
        """
        super(QNetwork, self).__init__()

        # ── Convolutional "feature extractor" ────────────────────────────────
        # These layers scan across the image, learning to recognise visual
        # patterns like the ball, the paddles, and their edges.
        #
        # Arguments: nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        #   in_channels  = how many input feature maps (= FRAME_STACK = 4 at first)
        #   out_channels = how many filters (feature maps) the layer produces
        #   kernel_size  = size of the sliding window (e.g. 8×8 pixels)
        #   stride       = how many pixels the window jumps each step
        #                  (larger stride = smaller output = faster computation)

        self.conv_layers = nn.Sequential(

            # Layer 1: 4 input channels (4 stacked frames) → 32 feature maps
            #   Kernel 8×8, stride 4 → output size ≈ 20×20
            nn.Conv2d(in_channels=FRAME_STACK, out_channels=32,
                      kernel_size=8, stride=4),
            nn.ReLU(),   # ReLU kills negative activations — adds non-linearity

            # Layer 2: 32 → 64 feature maps
            #   Kernel 4×4, stride 2 → output size ≈ 9×9
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=4, stride=2),
            nn.ReLU(),

            # Layer 3: 64 → 64 feature maps
            #   Kernel 3×3, stride 1 → output size ≈ 7×7
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # ── Fully connected "decision maker" ─────────────────────────────────
        # After the conv layers, we flatten the 64 feature maps (each 7×7)
        # into a single long vector: 64 × 7 × 7 = 3136 values.
        # Then two linear layers map that down to one Q-value per action.

        self.fc_layers = nn.Sequential(

            # 3136 inputs → 512 hidden units
            nn.Linear(in_features=64 * 7 * 7, out_features=512),
            nn.ReLU(),

            # 512 hidden units → one output per action (the Q-values)
            nn.Linear(in_features=512, out_features=num_actions),
            # NOTE: No activation at the end. Q-values can be any real number,
            # so we do NOT apply Softmax or ReLU here.
        )

    def forward(self, pixel_frames):
        """
        Forward pass: takes in pixel frames, returns Q-values.

        pixel_frames: tensor of shape (batch_size, FRAME_STACK, 84, 84)
                      with pixel values normalised to [0, 1].
        Returns:      tensor of shape (batch_size, num_actions)
        """
        # Pass through all three convolutional layers
        conv_output = self.conv_layers(pixel_frames)

        # Flatten: (batch_size, 64, 7, 7) → (batch_size, 3136)
        # batch_size stays intact; everything else collapses into one dimension.
        flat = conv_output.view(conv_output.size(0), -1)

        # Pass through the fully connected layers to get Q-values
        q_values = self.fc_layers(flat)

        return q_values


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — EXPERIENCE REPLAY BUFFER
#
# Why do we need this?
#   If we trained on each experience as it happens (online learning), two
#   problems occur:
#     1. Consecutive experiences are highly correlated — the agent just walked
#        through the same corridor for 50 steps. Training on correlated data
#        makes the network overfit to recent events and "forget" older ones.
#     2. We waste data — each experience is used exactly once then thrown away.
#
#   Solution: store every (state, action, reward, next_state, done) tuple in a
#   big circular buffer, then at training time RANDOMLY SAMPLE a mini-batch
#   of 32 experiences. Random sampling breaks the correlation between samples
#   and lets us reuse past experiences many times.
#
# What is stored in one experience (a "transition")?
#   state      — the stacked frames the agent saw BEFORE taking the action
#   action     — the integer action the agent chose (e.g. 2 = paddle up)
#   reward     — the scalar reward received (+1 scored, -1 conceded, 0 otherwise)
#   next_state — the stacked frames the agent saw AFTER taking the action
#   done       — boolean: True if this action ended the game (episode over)
#
# The buffer is implemented as a deque (double-ended queue) with a maximum
# length equal to BUFFER_CAPACITY. When the buffer is full and a new
# experience arrives, the OLDEST experience is automatically dropped.
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Circular buffer that stores past (state, action, reward, next_state, done)
    tuples and serves random mini-batches for training.
    """

    def __init__(self, capacity):
        """
        capacity: maximum number of experiences to store before old ones
                  are overwritten. E.g. 10_000.
        """
        # deque with maxlen automatically drops the oldest item when full.
        # This is our "memory" — it replaces experiences the agent had long ago.
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        """
        Save one experience to the buffer.

        state      — numpy array, shape (FRAME_STACK, 84, 84), dtype uint8
        action     — int, the action index chosen by the agent
        reward     — float, the reward received from the environment
        next_state — numpy array, same shape as state, the following observation
        done       — bool, True if the episode ended after this step
        """
        # Pack everything into a simple tuple and append to the buffer.
        # If the buffer is at capacity, deque automatically removes the oldest.
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Randomly draw `batch_size` experiences from the buffer.

        Returns five separate tensors (ready to feed straight into the network):
            states      — shape (batch_size, FRAME_STACK, 84, 84), float32 in [0,1]
            actions     — shape (batch_size,), int64
            rewards     — shape (batch_size,), float32
            next_states — shape (batch_size, FRAME_STACK, 84, 84), float32 in [0,1]
            dones       — shape (batch_size,), float32  (1.0 if done, 0.0 if not)
        """
        # Randomly pick `batch_size` experiences from wherever they are in the buffer
        batch = random.sample(self.buffer, batch_size)

        # The batch is a list of tuples. We "unzip" it into separate lists,
        # one list per field.
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert each list into a NumPy array first, then a PyTorch tensor.
        # --- States ---
        # Stack list of (FRAME_STACK, 84, 84) arrays → (batch_size, FRAME_STACK, 84, 84)
        # Divide by 255.0 to normalise pixel values from [0, 255] to [0.0, 1.0].
        # Neural networks learn much better with small input values near zero.
        states_tensor = torch.tensor(
            np.array(states), dtype=torch.float32
        ).to(DEVICE) / 255.0

        # --- Actions ---
        # int64 (long) is required by PyTorch's gather() function used during training
        actions_tensor = torch.tensor(
            np.array(actions), dtype=torch.int64
        ).to(DEVICE)

        # --- Rewards ---
        rewards_tensor = torch.tensor(
            np.array(rewards), dtype=torch.float32
        ).to(DEVICE)

        # --- Next States ---
        next_states_tensor = torch.tensor(
            np.array(next_states), dtype=torch.float32
        ).to(DEVICE) / 255.0

        # --- Dones ---
        # Convert True/False → 1.0/0.0 so we can use it in math during training.
        # (We will multiply by (1 - done) to zero out future rewards when the
        #  episode has ended — a done state has no future.)
        dones_tensor = torch.tensor(
            np.array(dones, dtype=np.float32), dtype=torch.float32
        ).to(DEVICE)

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor

    def __len__(self):
        """
        Lets us call len(replay_buffer) to check how many experiences are stored.
        We use this to make sure the buffer has enough samples before we start
        training (we do not train until len(buffer) >= BATCH_SIZE).
        """
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — EPSILON-GREEDY ACTION SELECTION
#
# The agent faces a fundamental dilemma every single step:
#   - EXPLOIT: pick the action with the highest Q-value (use what it knows)
#   - EXPLORE: pick a random action (discover something new)
#
# We balance this with an epsilon-greedy strategy:
#   - With probability epsilon        → take a RANDOM action (explore)
#   - With probability (1 - epsilon)  → take the BEST known action (exploit)
#
# Early in training epsilon = 1.0 (100% random). We decay it after every
# episode by multiplying by EPSILON_DECAY (e.g. 0.995), so the agent
# gradually shifts from pure exploration to mostly exploitation as it learns.
#
# We never let epsilon drop below EPSILON_END (e.g. 0.05) so the agent
# always keeps a small chance of trying something new — avoiding getting
# permanently stuck in a bad strategy.
# ─────────────────────────────────────────────────────────────────────────────

def select_action(state, online_network, epsilon, num_actions):
    """
    Choose an action using the epsilon-greedy strategy.

    state          — numpy array, shape (FRAME_STACK, 84, 84), the current observation
    online_network — the QNetwork used to estimate Q-values
    epsilon        — float in [0, 1], current exploration probability
    num_actions    — int, total number of possible actions

    Returns: int — the chosen action index
    """
    # Roll a random number between 0 and 1.
    # If it falls below epsilon → explore (random action).
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)   # Random action index

    # Otherwise → exploit: ask the network which action looks best.
    # We must:
    #   1. Convert the numpy state to a PyTorch tensor
    #   2. Add a batch dimension: (4, 84, 84) → (1, 4, 84, 84)
    #   3. Normalise pixels to [0, 1] by dividing by 255
    #   4. Turn off gradient tracking (we are not training here, just predicting)
    with torch.no_grad():
        state_tensor = torch.tensor(
            np.array(state), dtype=torch.float32
        ).unsqueeze(0).to(DEVICE) / 255.0          # shape: (1, 4, 84, 84)

        q_values = online_network(state_tensor)     # shape: (1, num_actions)

    # argmax picks the index of the highest Q-value → the greedy best action
    return q_values.argmax(dim=1).item()            # .item() converts tensor → Python int


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — OPTIMIZE MODEL (BELLMAN EQUATION + LOSS + BACKPROP)
#
# This is the mathematical heart of DQN. It runs once per step, after we have
# sampled a random batch from the replay buffer.
#
# --- What is the Bellman Equation? ---
# The Bellman equation defines the IDEAL Q-value for taking action `a`
# in state `s`:
#
#   Q*(s, a) = reward  +  gamma * max_a'[ Q*(next_state, a') ] * (1 - done)
#
# In plain English:
#   "The true value of an action equals the immediate reward PLUS the
#    discounted value of the best action in the next state."
#   The (1 - done) term zeros out the future reward if the episode ended —
#   a terminal state has no future.
#
# --- Why do we need a TARGET NETWORK? ---
# If we used the SAME network to both predict Q-values AND calculate the
# target Q-values, we'd be chasing a moving target — like trying to hit a
# bullseye that moves every time you shoot. Training becomes unstable.
#
# Solution: use a FROZEN copy of the network (the target_network) to calculate
# the target Q-values. Its weights are only updated every TARGET_UPDATE_FREQ
# episodes by copying the online_network's weights into it.
# This keeps the targets stable for a while, making learning much smoother.
#
# --- Loss Calculation ---
# We use Huber Loss (SmoothL1Loss) instead of plain MSE because it is less
# sensitive to large errors (outlier rewards), which are common early in training.
#
#   Loss = HuberLoss( Q_online(s, a),  target_Q )
#
# --- Backpropagation ---
# Standard PyTorch training loop:
#   1. Zero out old gradients  (optimizer.zero_grad)
#   2. Compute the loss
#   3. Backpropagate           (loss.backward)
#   4. Update network weights  (optimizer.step)
# ─────────────────────────────────────────────────────────────────────────────

def optimize_model(online_network, target_network, replay_buffer, optimizer):
    """
    Sample one batch from the buffer and perform one gradient update on the
    online network.

    online_network  — the QNetwork we are actively training
    target_network  — the frozen copy used to compute stable Q-value targets
    replay_buffer   — the ReplayBuffer to sample experiences from
    optimizer       — the Adam optimizer attached to the online_network

    Returns: float or None — the scalar loss value (or None if buffer has insufficient samples)
    """
    # Guard: do not train until we have enough experiences stored for at least one batch
    if len(replay_buffer) < BATCH_SIZE:
        return None

    # Pull a random mini-batch of 32 experiences from memory.
    # Each variable is already a GPU/CPU tensor (handled inside buffer.sample).
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    # ── Step 1: Compute PREDICTED Q-values from the online network ───────────
    # online_network(states) → shape (32, num_actions)
    # We only want the Q-value for the action that was ACTUALLY taken.
    # .gather(1, actions) picks one value per row using the action index.
    # Result shape: (32, 1) → we squeeze to (32,)
    predicted_q_values = online_network(states).gather(
        dim=1,
        index=actions.unsqueeze(1)   # shape (32,) → (32, 1) for gather
    ).squeeze(1)                     # (32, 1) → (32,)

    # ── Step 2: Compute TARGET Q-values using the Bellman equation ───────────
    # We use target_network (not online_network!) for stability.
    # torch.no_grad() means we do NOT compute gradients through this —
    # the target is treated as a fixed label, not something to optimise.
    with torch.no_grad():
        # Get the best Q-value achievable from the NEXT state.
        # .max(dim=1).values → shape (32,)  — the highest Q-value per sample
        best_next_q_values = target_network(next_states).max(dim=1).values

        # Bellman target: reward + gamma * max_Q(next_state) * (1 - done)
        # If done=1 (episode ended), the future term vanishes.
        target_q_values = rewards + GAMMA * best_next_q_values * (1.0 - dones)

    # ── Step 3: Compute the loss ──────────────────────────────────────────────
    # Huber Loss = smooth blend of L1 (robust to outliers) and L2 (smooth near 0).
    # It penalises large prediction errors less aggressively than MSE, which
    # helps when the agent occasionally gets wildly wrong Q-value estimates.
    loss_fn = nn.SmoothL1Loss()
    loss = loss_fn(predicted_q_values, target_q_values)

    # ── Step 4: Backpropagation ───────────────────────────────────────────────
    optimizer.zero_grad()   # Clear gradients from the previous step
    loss.backward()         # Compute new gradients via backpropagation
    optimizer.step()        # Update all weights in online_network

    return loss.item()      # Return the scalar loss for logging


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — MAIN TRAINING LOOP
#
# This is the outer loop that runs NUM_EPISODES full Pong games.
# Each episode:
#   1. Reset the environment to get the starting observation
#   2. Loop over steps: select action → step env → store experience → train
#   3. Decay epsilon after the episode ends
#   4. Every TARGET_UPDATE_FREQ episodes: copy online → target network weights
#   5. Print progress and save the model
#
# The relationship between all components:
#
#   make_env()           — provides observations and rewards
#   select_action()      — decides what to do (explore vs exploit)
#   replay_buffer.store()— remembers what happened
#   optimize_model()     — learns from past experiences
#   target_network       — provides stable Q-value targets during optimization
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Set up environment ────────────────────────────────────────────────────
    env = make_env()
    num_actions = env.action_space.n    # 6 for Pong

    # ── Set up networks ───────────────────────────────────────────────────────
    # The ONLINE network is trained every step — this is our main "brain".
    online_network = QNetwork(num_actions=num_actions).to(DEVICE)

    # The TARGET network starts as an exact copy of the online network.
    # Its weights are frozen and only updated every TARGET_UPDATE_FREQ episodes.
    target_network = QNetwork(num_actions=num_actions).to(DEVICE)
    target_network.load_state_dict(online_network.state_dict())
    target_network.eval()   # Set to eval mode — we never call .backward() on it

    # ── Set up optimizer ──────────────────────────────────────────────────────
    # Adam is a popular adaptive optimizer. Only the ONLINE network is optimised.
    optimizer = torch.optim.Adam(online_network.parameters(), lr=LEARNING_RATE)

    # ── Set up replay buffer ──────────────────────────────────────────────────
    replay_buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)

    # ── Epsilon starts fully random ───────────────────────────────────────────
    epsilon = EPSILON_START

    # ── Logging ───────────────────────────────────────────────────────────────
    episode_rewards = []    # Track total reward per episode to monitor progress
    episode_lengths = []    # Track number of steps per episode

    print(f"Starting training for {NUM_EPISODES} episodes on {DEVICE}...")
    print(f"Actions available: {num_actions}")
    print("-" * 50)

    # ── Episode loop ──────────────────────────────────────────────────────────
    for episode in range(1, NUM_EPISODES + 1):

        # Reset environment to the beginning of a new Pong game.
        # obs shape: (4, 84, 84) — 4 stacked greyscale frames
        obs, _ = env.reset()
        state = np.array(obs)       # Convert to numpy array for processing

        episode_reward = 0.0        # Accumulate reward over the whole episode
        episode_loss   = 0.0        # Accumulate loss for logging
        step_count     = 0          # Count steps in this episode

        # ── Step loop (one full Pong game) ────────────────────────────────────
        done = False
        while not done:

            # 1. Choose an action using epsilon-greedy
            action = select_action(state, online_network, epsilon, num_actions)

            # 2. Take the action in the environment
            # Gymnasium returns: next_obs, reward, terminated, truncated, info
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_obs)

            # An episode ends if the game finishes (terminated) OR
            # if it hits a time limit (truncated)
            done = terminated or truncated

            # 3. Store this transition in the replay buffer
            replay_buffer.store(state, action, reward, next_state, done)

            # Move to the next state
            state = next_state
            episode_reward += reward
            step_count += 1

            # 4. Train the model (optimize_model internally checks if buffer has enough samples)
            loss = optimize_model(online_network, target_network,
                                  replay_buffer, optimizer)
            if loss is not None:
                episode_loss += loss

        # ── End of episode ────────────────────────────────────────────────────
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)

        # Decay epsilon: multiply by EPSILON_DECAY, but never go below EPSILON_END.
        # This gradually shifts the agent from exploration → exploitation.
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Every TARGET_UPDATE_FREQ episodes, copy the online network's weights
        # into the target network. This "re-anchors" the Bellman targets.
        if episode % TARGET_UPDATE_FREQ == 0:
            target_network.load_state_dict(online_network.state_dict())

        # Print a progress summary every 10 episodes
        if episode % 1 == 0:
            avg_reward = np.mean(episode_rewards[-10:])  # Average of last 10
            avg_loss   = episode_loss / max(step_count, 1)
            print(f"Episode {episode:4d}/{NUM_EPISODES} | "
                  f"Avg Reward (last 10): {avg_reward:6.1f} | "
                  f"Epsilon: {epsilon:.3f} | "
                  f"Avg Loss: {avg_loss:.4f} | "
                  f"Buffer: {len(replay_buffer)}")

    # ── Save the trained model weights ────────────────────────────────────────
    # We only save the ONLINE network — that is the trained brain.
    # play.py will load these weights to reconstruct the network for evaluation.
    torch.save(online_network.state_dict(), SAVE_PATH)
    print("-" * 50)
    print(f"Training complete! Model saved to '{SAVE_PATH}'")

    # ── Generate and Save Training Plots ──────────────────────────────────────
    print("Generating training plots...")
    plot_path = SAVE_PATH.replace('.pth', '.png')  # e.g., dqn_model_exp1.png
    
    plt.figure(figsize=(12, 5))
    
    # Left Graph: Rewards
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, color="blue", alpha=0.7)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True, alpha=0.3)
    
    # Right Graph: Steps
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

# ─────────────────────────────────────────────────────────────────────────────
# Entry point: only run main() when this file is executed directly,
# not when it is imported as a module (e.g. during smoke tests).
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()