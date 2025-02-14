#!/bin/sh
env="Aps"
algo="happo"
exp="updated_power_apconst05/ped/happo"
seed=1
n_rollout_threads=16

python train/train_aps.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
 --seed ${seed} --n_training_threads 16 --n_rollout_threads ${n_rollout_threads} \
 --num_mini_batch 1 --episode_length 100 --num_env_steps 160000 --ppo_epoch 5 \
 --lr 7e-4 --critic_lr 1e-3 --entropy_coef 0.015 --gamma 0.01 \
 --max_grad_norm 1 --log_interval 1