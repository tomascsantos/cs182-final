# Phasic Policy Gradient

#### [[Paper]](https://arxiv.org/abs/2009.04416)

This is code for training agents using [Phasic Policy Gradient](https://arxiv.org/abs/2009.04416) [(citation)](#citation).

Supported platforms:

- macOS 10.14 (Mojave)
- Ubuntu 16.04

Supported Pythons:

- 3.7 64-bit

## Install

We have been running on Google Colab with the following requirements:
!pip install mpi4py==3.0.3 gym3==0.3.3 procgen==0.10.4 pandas==1.1.1 matplotlib==3.1.1 tensorflow==1.15.0

It comes preinstalled with torch==1.8.1+cu101, but torch==1.4.0 also works and versions between.

In addition one can use conda and the environment.yml file located in the repo.  We included a .ipynb for colab use that contains everything needed to run

## Reproduce and Visualize Results
python -m rl_project.train --log_dir [logging directory name] --num_levels [number of training levels, default=100] --data_aug [gray, cutout, cutout_color, flip, rotate, color_jitter, crop, None, default=None]



## Logging Name Syntax
Most of the runs are named using the following convention ppg\_{num envs}\_{num eval levels}\_{num training levels}\_{optional: data augmentation}
So the run ppg\_16\_all\_50\_crop is ppg run with the crop augmentation run on 50 training levels and infinite evaluation levels starting at 50.  So training is done on levels [0, 49] and evaluation is on [50, infinity]

In csv logs the key columns are: B: EpLenMean (Average length of an episode), C: EpRewMean (Average reward in training episodes), I: Misc/InteractCount (Number of environment steps), AG: EvalEpMean (Average reward in evaluation Episodes, collected less frequently)


##Note
The best_test_log is from ppg_16_all_50_cutout_color and test_run is the last column
