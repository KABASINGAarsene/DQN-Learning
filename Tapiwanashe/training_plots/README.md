# Tapiwanashe training plots

`exp_01.png` through `exp_10.png` show training rewards and episode lengths for the matching model numbers. The orange line, where present, is the mean reward over the last ten completed games. Interrupted or budget-ended games are excluded from completed-game curves.

`evaluation_comparison.png` compares mean rewards from ten greedy games per saved model. Training rewards include exploration; evaluation rewards do not. Different training methods and recorded budgets limit direct comparisons.

The exact settings, evaluation scores and saved model budgets are recorded in the matching files under `../models/`. The individual hyperparameter table is in the repository's `Training Experiment Tables` folder.
