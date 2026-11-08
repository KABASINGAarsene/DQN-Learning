"""
play.py - Evaluation Script for Deep Q-Network (DQN) Atari Pong
===============================================================
Assignment: Formative 2 — Evaluation (play.py) Rubric Alignment (5 Points)

This script loads the trained weights from 'dqn_model.pth', rebuilds the
CNN architecture, and plays Pong live with visual rendering enabled.

Key Requirements Demonstrated:
  1. Loads 'dqn_model.pth' checkpoint.
  2. Reconstructs identical QNetwork architecture.
  3. Sets model.eval() mode (disables gradient tracking / dropout).
  4. Uses 100% GREEDY action selection (epsilon = 0, no random exploration).
  5. Live renders the Pong environment using Gymnasium render_mode='human'.
"""

import time
import os
import numpy as np
import torch
import gymnasium as gym
import ale_py

# Register ALE environments so Gymnasium recognizes 'ALE/Pong-v5'
gym.register_envs(ale_py)

# Import the identical QNetwork class and hyperparameters from train.py
# to guarantee architectural consistency.
from train import QNetwork, DEVICE, FRAME_STACK, ENV_NAME, SAVE_PATH


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — EVALUATION ENVIRONMENT (WITH LIVE RENDERING)
#
# Unlike training (where render_mode=None for maximum speed), evaluation
# requires render_mode="human" so we can visually watch the agent play.
#
# We apply the EXACT same observation preprocessing pipeline as train.py:
#   1. ResizeObservation  -> downsample to 84x84 pixels
#   2. GrayscaleObservation -> drop RGB channels (keep_dim=False)
#   3. FrameStackObservation -> stack 4 consecutive frames for velocity
# ─────────────────────────────────────────────────────────────────────────────

def make_eval_env():
    """
    Creates and returns the evaluation environment with live window rendering.
    Observation shape: (FRAME_STACK, 84, 84)
    """
    # render_mode="human" pops open a live display window showing the game
    env = gym.make(ENV_NAME, render_mode="human")

    # Resize to 84x84
    env = gym.wrappers.ResizeObservation(env, shape=(84, 84))

    # Convert RGB to single-channel greyscale
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)

    # Stack the last 4 frames so the agent can perceive ball velocity
    env = gym.wrappers.FrameStackObservation(env, stack_size=FRAME_STACK)

    return env


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — GREEDY ACTION SELECTION (EXPLOITATION ONLY)
#
# During evaluation, exploration is completely disabled (epsilon = 0.0).
# The agent must only pick the action with the HIGHEST predicted Q-value.
# ─────────────────────────────────────────────────────────────────────────────

def select_greedy_action(state, model):
    """
    Selects the best action strictly based on the model's Q-values.

    state: numpy array of shape (4, 84, 84)
    model: trained QNetwork in eval() mode

    Returns: int — best action index
    """
    with torch.no_grad():
        # Convert state to tensor: (4, 84, 84) -> (1, 4, 84, 84), normalized to [0, 1]
        state_tensor = torch.tensor(
            np.array(state), dtype=torch.float32
        ).unsqueeze(0).to(DEVICE) / 255.0

        # Predict Q-values for all possible actions
        q_values = model(state_tensor)

        # Pick the action with the maximum Q-value
        best_action = q_values.argmax(dim=1).item()

    return best_action


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — MAIN EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Verify trained weights exist ──────────────────────────────────────────
    if not os.path.exists(SAVE_PATH):
        print(f"Error: Trained model file '{SAVE_PATH}' not found!")
        print("Please run train.py first to produce 'dqn_model.pth'.")
        return

    # ── Create the rendering environment ──────────────────────────────────────
    print(f"Initializing live environment: {ENV_NAME} ...")
    env = make_eval_env()
    num_actions = env.action_space.n

    # ── Reconstruct Q-Network architecture & load weights ─────────────────────
    print(f"Loading model weights from '{SAVE_PATH}' onto {DEVICE}...")
    model = QNetwork(num_actions=num_actions).to(DEVICE)

    # Load the learned parameter weights into the architecture
    checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)

    # Set to EVALUATION mode.
    # This disables training-specific behaviors (e.g., dropout if present)
    # and ensures weights are treated as frozen.
    model.eval()
    print("Model loaded successfully in eval() mode.\n")

    # ── Play evaluation games ─────────────────────────────────────────────────
    TOTAL_GAMES = 3     # Number of full Pong matches to play live

    for game in range(1, TOTAL_GAMES + 1):
        obs, _ = env.reset()
        state = np.array(obs)

        total_score = 0.0
        step_count = 0
        done = False

        print(f"--- Starting Match {game}/{TOTAL_GAMES} ---")

        while not done:
            # Pick the greedy best action (no random exploration)
            action = select_greedy_action(state, model)

            # Step the environment forward
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_obs)

            done = terminated or truncated
            total_score += reward
            step_count += 1
            state = next_state

            # Pacing: sleep 0.02s (~50 fps) so the game plays at watchable human speed
            time.sleep(0.02)

        print(f"Match {game} finished! Final Score: {total_score:+.0f} after {step_count} steps\n")

    # Clean up window
    env.close()
    print("Evaluation completed.")


if __name__ == "__main__":
    main()

