### Hyperparameter Experiment Results

| **Experiment** | **Learning rate** | **Gamma** | **Batch size** | **Epsilon start** | **Epsilon end** | **Epsilon decay** | **Observed behaviour** |
|---|---:|---:|---:|---:|---:|---|---|
| **1 (Baseline)** | 0.001 | 0.99 | 32 | 1.0 | 0.05 | 0.995 | **Baseline failure:** After 500 episodes, greedy evaluation averaged −21. RIGHT was selected on every decision. |
| **2** | 0.001 | 0.99 | 32 | 1.0 | 0.05 | 0.999 | **Slower decay did not help:** After 500 episodes, evaluation stayed at −21. FIRE was selected throughout. |
| **3** | 0.00025 | 0.99 | 32 | 1.0 | 0.05 | 0.999 | **Lower learning rate did not resolve failure:** Evaluation stayed at −21 with FIRE throughout. Diagnostics found inactive final convolution features on sampled states. |
| **4** | 0.00005 | 0.99 | 32 | 1.0 | 0.05 | 0.999 | **Learning improved:** Lowering the learning rate to 0.00005 gave an evaluation mean of −14.5 after 500 episodes. The agent used several actions. |
| **5** | 0.00005 | 0.99 | 32 | 1.0 | 0.05 | 0.995 | **Faster decay improved this run:** Evaluation mean rose to −8.9 after 500 episodes, compared with −14.5 in experiment 4. No evaluation games were won. |
| **6** | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | Linear: 100,000 steps | **Revised training setup:** Added replay warm-up, step-based scheduling and less frequent learning. Evaluation at 511,428 steps gave an evaluation mean −15.6. Several settings changed together. |
| **7** | 0.0001 | 0.995 | 32 | 1.0 | 0.01 | Linear: 100,000 steps | **Best evaluation performance:** With gamma 0.995, the revised setup completed 1,000,000 steps. Evaluation mean +4.1; won 9 of 10 games. |
| **8** | 0.0001 | 0.99 | 32 | 0.5 | 0.01 | Linear: 100,000 steps | **Constant-action failure:** With post-warm-up epsilon starting at 0.5, evaluation at 850,000 steps averaged −21 and selected LEFT throughout. |
| **9** | 0.00005 | 0.99 | 64 | 1.0 | 0.05 | 0.995 | **Larger batch did not improve this run:** Using the original method, batch size 64 completed 500 episodes. Evaluation mean −10.6, compared with −8.9 for batch size 32 in experiment 5. |
| **10** | 0.00005 | 0.995 | 32 | 1.0 | 0.05 | 0.995 | **Higher gamma did not improve this run:** With the original training method, gamma 0.995 completed 500 episodes. Evaluation mean −10.5, compared with −8.9 for gamma 0.99 in experiment 5. No evaluation games were won. |

Experiments 1–5, 9 and 10 use the original 500-episode method. Experiments 6–8 use the revised step-based method. In experiments 6–8, the first 10,000 steps are random; linear epsilon decay follows the warm-up. Their other settings are: train every 4 steps, target update every 1,000 steps, and replay capacity 50,000. These changes mean comparisons between the two methods do not isolate one hyperparameter.

All evaluation means use ten games without random exploration. Results describe individual runs; they do not prove that one hyperparameter alone caused the differences.
