# Deep Q-Network (DQN) - Atari Pong

**Assignment:** Formative 2 — Implementing a DQN agent from scratch.

This repository contains the baseline implementation of a DQN agent trained to play `ALE/Pong-v5`. The codebase is divided into two main scripts:

* `train.py`: Contains the environment setup, CNN architecture, Replay Buffer, and training loop.

* `play.py`: The evaluation script that loads a trained model and plays the game live.

##  Local Setup (Windows & macOS)

We are using `uv` to manage our dependencies to ensure everyone has the exact same versions of PyTorch and Gymnasium.

Follow these steps to set up the environment on your local machine:

**1. Clone the repository**
Navigate to your desired folder in your terminal and clone this repo.

**2. Install the `uv` package manager**

* **Mac / Linux / Windows (Git Bash):**
  `curl -LsSf https://astral.sh/uv/install.sh | sh`

* *(Restart your terminal after installing)*

**3. Build the environment**
Run the following command in the project root. This reads the `uv.lock` file and installs all identical dependencies:
`uv sync`

**4. Activate the virtual environment**

* **Windows (Git Bash):** `source .venv/Scripts/activate`

* **macOS / Linux:** `source .venv/bin/activate`

**5. Verify the setup**
Run the baseline training script. If it starts printing episode progress without crashing, your local setup is perfect.
`python train.py`

##  Running Experiments on Kaggle (GPUs)

Training 500 episodes of Pong locally on a CPU takes 1-3 hours. For our hyperparameter tuning experiments, **we must use Kaggle** to access free NVIDIA GPUs, which reduces training time to about 15-20 minutes.

**Kaggle Step-by-Step Workflow:**

1. Open a new Kaggle Notebook.

2. In the right-hand panel, go to **Notebook options > Accelerator** and select **GPU T4 x2**.

3. In the very first cell, run this command to install the Atari ROMs in the cloud:
   `!pip install "gymnasium[atari,accept-rom-license]" ale-py`

4. Copy the entire contents of `train.py` and paste it into the next cell.

5. **CRITICAL:** At the top of the code, change the `SAVE_PATH` variable for your specific experiment so you don't overwrite previous models (e.g., `SAVE_PATH = "dqn_model_exp1.pth"`).

6. Tweak the hyperparameters (Learning Rate, Gamma, Epsilon Decay, etc.) as needed for your experiment.

7. Run the cell. When it finishes 500 episodes, download your specific `.pth` file from the `/kaggle/working/` output directory.

##  Hyperparameter Tuning Log

Each team member is responsible for running 10 experiments. Please log your results in the table below so we can find the best combination of hyperparameters.

**Hardware Used:** Kaggle T4 GPU

| Experiment | Learning Rate | Gamma | Epsilon Decay | Target Update Freq | Batch Size | Avg Reward (Last 10) | Notes / Observations | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **Baseline** | 0.001 | 0.99 | 0.995 | 10 | 32 | \[Score\] | Default settings. Agent struggled to track the ball. | 
| **Exp 1** |  |  |  |  |  |  |  | 
| **Exp 2** |  |  |  |  |  |  |  | 
| **Exp 3** |  |  |  |  |  |  |  | 
| **Exp 4** |  |  |  |  |  |  |  | 
| **Exp 5** |  |  |  |  |  |  |  | 
| **Exp 6** |  |  |  |  |  |  |  | 
| **Exp 7** |  |  |  |  |  |  |  | 
| **Exp 8** |  |  |  |  |  |  |  | 
| **Exp 9** |  |  |  |  |  |  |  | 
| **Exp 10** |  |  |  |  |  |  |  | 

### Final Conclusion

*(After completing the experiments, summarize which combination performed best and why.)*