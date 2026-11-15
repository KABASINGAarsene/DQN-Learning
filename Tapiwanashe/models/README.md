# Tapiwanashe models

Each experiment has three matching files: `.pth` weights, `.config.json` training settings and `.evaluation.json` evaluation evidence. The SHA-256 in each evaluation record identifies the exact saved model.

Evaluation used greedy action selection over seeds 10000–10009. Saved results were checked for arithmetic consistency; training and evaluation were not rerun for these commits.

| Experiment | Method | Recorded training steps | Mean evaluation reward |
|---|---|---:|---:|
| 1 | original | 434,090 | -21.0 |
| 2 | original | 455,765 | -21.0 |
| 3 | original | 448,339 | -21.0 |
| 4 | original | 560,494 | -14.5 |
| 5 | original | 873,237 | -8.9 |
| 6 | step_based | 511,428 | -15.6 |
| 7 | step_based | 1,000,000 | +4.1 |
| 8 | step_based | 850,000 | -21.0 |
| 9 | original | 851,837 | -10.6 |
| 10 | original | 865,506 | -10.5 |

Experiments 1–5, 9 and 10 used 500 episodes. Experiments 6–8 used a step-based procedure; the table reports the actual saved model budget. Differences between the two procedures change several training settings at once.

Experiment 7 has the best recorded evaluation: +4.1 and 9 wins in 10 games. Its weights are `exp_07.pth`.

The group gameplay script and training script are unchanged by this evidence branch.
