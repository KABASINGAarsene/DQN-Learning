# Deep Q-Network (DQN) - Atari Pong

## 1. Project Introduction
* **Group Members:** [Kabasinga Arsene], [Carla Liza Batoni], [Tapiwanashe  Gift Marufu],
* **Selected Atari Environment:** `ALE/Pong-v5`

This project implements a Deep Q-Network (DQN) agent from scratch using PyTorch and Gymnasium. The agent was trained to play Atari Pong by processing raw pixel inputs, breaking the correlation of sequential data through experience replay, and balancing exploration with exploitation to maximize cumulative expected rewards.

---

## 2. DQN Architecture

### Network Diagram
Our CNN architecture is designed to map raw pixel states to action values. It takes a stack of 4 grayscale frames as input to capture temporal dynamics (motion and velocity), processes them through three convolutional layers to extract spatial features, and outputs the predicted Q-values for all 6 possible paddle actions in the Pong environment.

```text
[Input] Stacked Frames: 4 channels (84x84 pixels)
                 │
                 ▼
[Conv2D Layer 1] 32 filters, 8x8 kernel, stride 4, ReLU activation
                 │
                 ▼
[Conv2D Layer 2] 64 filters, 4x4 kernel, stride 2, ReLU activation
                 │
                 ▼
[Conv2D Layer 3] 64 filters, 3x3 kernel, stride 1, ReLU activation
                 │
                 ▼
          [Flatten Layer]
                 │
                 ▼
[Fully Connected Layer 1] 512 nodes, ReLU activation
                 │
                 ▼
[Fully Connected Layer 2] 6 Output nodes (1 per action) -> Q-Values

## Explanation of the Training Process

Our training loop is built on several foundational Reinforcement Learning mechanisms to ensure stable and convergent learning:

* **Environment Preprocessing:** To reduce the computational load and state space, the raw game screen (`210x160` RGB) is converted to grayscale, downsampled to `84x84` pixels, and stacked in batches of 4 consecutive frames. This provides the Markov Decision Process (MDP) with the necessary context to determine ball trajectory and velocity.
* **Epsilon-Greedy Action Selection:** The agent balances discovering new strategies (exploration) and leveraging known rewarding actions (exploitation). We implemented a decay strategy where the exploration probability ($\epsilon$) decays exponentially over time from a starting value of `1.0` down to a minimum threshold of `0.02`. This forces the agent to rely on its trained neural network in the later stages of training.
* **Experience Replay Buffer:** Reinforcement learning updates are highly unstable if trained on sequential, correlated frames. Every step `(state, action, reward, next state, done flag)` is saved in a memory buffer of `10,000` limits. The agent samples random mini-batches of experiences from this buffer to train on, breaking the i.i.d. assumption and smoothing out the data distribution.
* **Optimization & The Bellman Equation:** The agent uses a dual-network setup to stabilize learning. The "Online Network" predicts the Q-value of the current state. A separate, frozen "Target Network" calculates the expected future reward using the Bellman equation:
  
  $$Q_{target} = r + \gamma \max_{a'} Q_{target}(s', a')$$
  
  The loss is computed between the online network's prediction and this target using a loss function, and the Adam optimizer performs backpropagation to continuously update the online network's weights.
* **Target Network Updates:** To prevent the expected reward target from shifting constantly while the agent tries to learn it, the Target Network is frozen and only receives the updated weights from the Online Network periodically. This provides a stable, fixed objective for the optimizer during those intervals.

## 3. Gameplay Demonstration


Below In the demonstration, our trained agent controls the green paddle on the righti playing Pong after being trained with an optimized learning rate of `0.0001` and an aggressive epsilon decay of `0.950` to force early exploitation.
[GAME PLAY VIDEO](https://youtu.be/o2ZNIaYt9uU).
